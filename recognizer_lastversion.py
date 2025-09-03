import cv2
import numpy as np
from mtcnn import MTCNN
from keras_facenet import FaceNet
import mysql.connector
import time
import tkinter as tk
from tkinter import messagebox
from sklearn.metrics.pairwise import cosine_similarity
import warnings
import os
import glob
from pathlib import Path
import json
import threading
from dotenv import load_dotenv
warnings.filterwarnings('ignore')

load_dotenv()

class OptimizedFaceRecognizerWithPhotos:
    def __init__(self):
        # Veritabanı bağlantısı
        self.conn = mysql.connector.connect(
            host=os.environ.get('DB_HOST', 'localhost'),
            user=os.environ.get('DB_USER', 'root'),
            password=os.environ.get('DB_PASSWORD', ''),
            database=os.environ.get('DB_NAME', 'face_recognition')
        )
        self.cursor = self.conn.cursor()
        
        # Modeller - tek seferlik yükle
        print("Models loading...")
        self.detector = MTCNN()
        self.embedder = FaceNet()
        
        # Optimize edilmiş threshold değerleri
        self.base_threshold = 0.7
        # Cosine similarity için eşik (>= 0.6 kabul)
        self.similarity_threshold = 0.6
        
        # Performans için cache
        self.face_cache = []
        self.embedding_cache = {}  # Photo path için embedding cache
        self.cache_size = 5
        
        # Validation parametreleri
        self.required_matches = 3
        self.consecutive_matches = 0
        # Düşük benzerlik reddi parametreleri
        self.low_similarity_threshold = 0.4
        self.low_similarity_required = 5
        self.consecutive_low_similarity = 0
        
        # Frame atlama
        self.frame_skip = 3
        self.frame_count = 0
        
        # Yüz algılama optimizasyonu
        self.last_face_box = None
        self.face_lost_frames = 0
        
        # Embedding yükleme durumu
        self.loading_embeddings = False
        self.known_embeddings_ready = False
        self.known_embeddings = None
        self.max_photos_per_person = 30
        self.max_embeddings = 15
        
        # Disk cache klasörü
        self.embeddings_cache_dir = os.path.join(os.path.dirname(__file__), 'embeddings_cache')
        try:
            os.makedirs(self.embeddings_cache_dir, exist_ok=True)
        except Exception:
            pass
        
    def load_person_embeddings(self, photo_paths):
        """Kişinin fotoğraflarından embedding'leri yükle"""
        embeddings = []
        
        for photo_path in photo_paths:
            if len(embeddings) >= self.max_embeddings:
                break
            # Cache kontrol et
            if photo_path in self.embedding_cache:
                embeddings.append(self.embedding_cache[photo_path])
                continue
                
            try:
                # Fotoğrafı yükle
                if not os.path.exists(photo_path):
                    print(f"Warning: Photo not found: {photo_path}")
                    continue
                    
                image = cv2.imread(photo_path)
                if image is None:
                    print(f"Warning: Could not load image: {photo_path}")
                    continue
                
                # Büyük görselleri küçült (hız için)
                h, w = image.shape[:2]
                max_side = max(h, w)
                scale = 1.0
                resized = image
                if max_side > 800:
                    scale = 800.0 / max_side
                    resized = cv2.resize(image, (int(w*scale), int(h*scale)))
                
                # Yüz algıla (MTCNN RGB bekler)
                rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                faces = self.detector.detect_faces(rgb_image)
                if not faces:
                    print(f"Warning: No face detected in: {photo_path}")
                    continue
                
                # En güvenilir yüzü al
                best_face = max(faces, key=lambda x: x['confidence'])
                if best_face['confidence'] < 0.9:  # Yüksek güven eşiği
                    print(f"Warning: Low confidence face in: {photo_path}")
                    continue
                
                # Yüzü çıkar ve embedding oluştur
                x, y, w, h = best_face['box']
                # Koordinatlar resized'a göredir, direkt oradan kırp
                face = resized[y:y+h, x:x+w]
                face = cv2.resize(face, (160, 160))
                
                embedding = self.embedder.embeddings([face])[0]
                # L2 normalize et
                norm = np.linalg.norm(embedding) + 1e-10
                embedding = embedding / norm
                embeddings.append(embedding)
                
                # Cache'e ekle
                self.embedding_cache[photo_path] = embedding
                
            except Exception as e:
                print(f"Error processing {photo_path}: {e}")
                continue
        
        return embeddings

    def _cache_paths(self, tc_kimlik):
        base = os.path.join(self.embeddings_cache_dir, f"{tc_kimlik}")
        return base + ".npy", base + ".json"

    def try_load_embeddings_cache(self, tc_kimlik, photo_paths):
        npy_path, meta_path = self._cache_paths(tc_kimlik)
        try:
            if not (os.path.exists(npy_path) and os.path.exists(meta_path)):
                return None
            with open(meta_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)
            cached = meta.get('photos', [])
            # Doğrulama: aynı dosyalar ve mtimelar eşleşmeli (örneklenmiş set için)
            current = [{
                'path': p,
                'mtime': os.path.getmtime(p) if os.path.exists(p) else None
            } for p in photo_paths]
            if len(cached) != len(current):
                return None
            for a, b in zip(cached, current):
                if a['path'] != b['path'] or abs(a['mtime'] - (b['mtime'] or -1)) > 1e-6:
                    return None
            arr = np.load(npy_path)
            return [emb for emb in arr]
        except Exception:
            return None

    def save_embeddings_cache(self, tc_kimlik, embeddings, photo_paths):
        npy_path, meta_path = self._cache_paths(tc_kimlik)
        try:
            np.save(npy_path, np.array(embeddings))
            meta = {
                'photos': [{
                    'path': p,
                    'mtime': os.path.getmtime(p) if os.path.exists(p) else None
                } for p in photo_paths]
            }
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(meta, f)
        except Exception:
            pass
    
    def get_photo_paths_from_db(self, tc_kimlik):
        """Veritabanından isim al; fotoğrafları known_faces dizininden topla."""
        self.cursor.execute("SELECT name, IFNULL(surname,'') FROM patients WHERE tc_kimlik = %s", (tc_kimlik,))
        row = self.cursor.fetchone()
        if not row:
            return []
        person_name, person_surname = row
        display_name = f"{person_name} {person_surname}".strip()

        photo_paths = []
        base_dir = r"C:\Users\Ezgi\Desktop\Python_repo\face_tanima\known_faces"
        person_folder = os.path.join(base_dir, display_name)  # folder name = "Name Surname"
        if not os.path.isdir(person_folder):
            # fallback: try just name folder
            person_folder = os.path.join(base_dir, person_name)

        if os.path.isdir(person_folder):
            for ext in ('*.jpg', '*.jpeg', '*.png', '*.bmp'):
                photo_paths.extend(glob.glob(os.path.join(person_folder, ext)))
                photo_paths.extend(glob.glob(os.path.join(person_folder, ext.upper())))
        return photo_paths

    def sample_photo_paths(self, photo_paths):
        """Aşırı fotoğrafı kısaltmak için dengeli örnekleme uygula"""
        if not photo_paths:
            return []
        if len(photo_paths) <= self.max_photos_per_person:
            return photo_paths
        indices = np.linspace(0, len(photo_paths) - 1, self.max_photos_per_person).astype(int)
        return [photo_paths[i] for i in indices]
    
    def preprocess_frame_fast(self, frame):
        """Hızlı görüntü ön işleme"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        
        if brightness < 80:
            gamma = 1.5
            table = np.array([((i / 255.0) ** (1.0/gamma)) * 255
                             for i in np.arange(0, 256)]).astype("uint8")
            frame = cv2.LUT(frame, table)
        
        return frame
    
    def detect_face_optimized(self, frame):
        """Optimize edilmiş yüz algılama"""
        h, w = frame.shape[:2]
        
        if self.last_face_box is not None:
            x, y, fw, fh = self.last_face_box
            
            margin = 0.3
            roi_x = max(0, int(x - fw * margin))
            roi_y = max(0, int(y - fh * margin))
            roi_w = min(w - roi_x, int(fw * (1 + 2*margin)))
            roi_h = min(h - roi_y, int(fh * (1 + 2*margin)))
            
            roi = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
            
            try:
                # ROI'yi RGB'ye çevirerek MTCNN'e ver
                roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                results = self.detector.detect_faces(roi_rgb)
                if results and results[0]['confidence'] > 0.8:
                    result = results[0]
                    box = result['box']
                    box[0] += roi_x
                    box[1] += roi_y
                    self.last_face_box = box
                    self.face_lost_frames = 0
                    return [result]
                else:
                    self.face_lost_frames += 1
            except:
                self.face_lost_frames += 1
        
        if self.last_face_box is None or self.face_lost_frames > 5:
            try:
                scale = 0.5
                small_frame = cv2.resize(frame, (int(w*scale), int(h*scale)))
                
                # Küçük frame'i RGB'ye çevirerek MTCNN'e ver
                small_rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                results = self.detector.detect_faces(small_rgb)
                if results:
                    best_result = max(results, key=lambda x: x['confidence'])
                    if best_result['confidence'] > 0.7:
                        box = best_result['box']
                        box = [int(coord / scale) for coord in box]
                        best_result['box'] = box
                        
                        self.last_face_box = box
                        self.face_lost_frames = 0
                        return [best_result]
            except Exception as e:
                print(f"Face detection error: {e}")
        
        return []
    
    def extract_face_embedding_fast(self, frame, face_box):
        """Hızlı embedding çıkarma"""
        x, y, w, h = face_box
        
        x = max(0, x)
        y = max(0, y)
        w = min(w, frame.shape[1] - x)
        h = min(h, frame.shape[0] - y)
        
        if w <= 0 or h <= 0 or w < 50 or h < 50:
            return None
        
        try:
            face = frame[y:y+h, x:x+w]
            face = cv2.resize(face, (160, 160))
            
            # Cache kontrol et
            for cached_face, cached_embedding in self.face_cache:
                if np.mean((cached_face - face) ** 2) < 1000:
                    return cached_embedding
            
            embedding = self.embedder.embeddings([face])[0]
            # L2 normalize et
            norm = np.linalg.norm(embedding) + 1e-10
            embedding = embedding / norm
            
            if len(self.face_cache) >= self.cache_size:
                self.face_cache.pop(0)
            self.face_cache.append((face, embedding))
            
            return embedding
            
        except Exception as e:
            print(f"Embedding error: {e}")
            return None
    
    def is_match_with_multiple_embeddings(self, known_embeddings, current_embedding):
        """Birden fazla embedding ile eşleşme kontrolü"""
        if not known_embeddings:
            return False, 0.0
        
        max_similarity = 0.0
        
        for known_embedding in known_embeddings:
            try:
                # current_embedding'i de normalize varsayalım
                current_norm = current_embedding / (np.linalg.norm(current_embedding) + 1e-10)
                similarity = cosine_similarity([known_embedding], [current_norm])[0][0]
                max_similarity = max(max_similarity, similarity)
            except Exception as e:
                print(f"Similarity calculation error: {e}")
                continue
        
        is_match = max_similarity >= self.similarity_threshold
        return is_match, max_similarity
    
    def assess_frame_quality_fast(self, frame):
        """Hızlı kalite değerlendirmesi"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        
        if brightness > 100:
            return 'good'
        elif brightness > 60:
            return 'medium'
        else:
            return 'poor'
    
    def run_recognition_optimized(self, tc_kimlik_input):
        """Optimize edilmiş ana tanıma işlemi"""
        # Veritabanından kişi bilgilerini al
        self.cursor.execute("SELECT id, name, surname FROM patients WHERE tc_kimlik = %s", (tc_kimlik_input,))
        result = self.cursor.fetchone()
        
        if not result:
            print("No person with this ID number was found..")
            return False
        
        person_id, person_name, person_surname = result
        full_name = f"{person_name} {person_surname}".strip()
        
        # Fotoğraf yollarını al ve örnekle
        photo_paths = self.get_photo_paths_from_db(tc_kimlik_input)
        if not photo_paths:
            print("No photos found for this person..")
            return False
        print(f"Found {len(photo_paths)} photos for {full_name}")
        sampled_paths = self.sample_photo_paths(photo_paths)
        if len(sampled_paths) < len(photo_paths):
            print(f"Sampling photos: using {len(sampled_paths)} of {len(photo_paths)}")
        
        # Önce disk cache dene
        cached = self.try_load_embeddings_cache(tc_kimlik_input, sampled_paths)
        if cached:
            print(f"Loaded {len(cached)} embeddings from cache")
            self.known_embeddings = cached
            self.known_embeddings_ready = True
            self.loading_embeddings = False
        else:
            # Embedding'leri arka planda yükle
            self.loading_embeddings = True
            self.known_embeddings_ready = False
            self.known_embeddings = None
            
            def _load_embeddings():
                try:
                    emb = self.load_person_embeddings(sampled_paths)
                    self.known_embeddings = emb
                    self.known_embeddings_ready = True
                    print(f"Created {len(emb)} embeddings from photos")
                    # Cache'e yaz
                    if emb:
                        self.save_embeddings_cache(tc_kimlik_input, emb, sampled_paths)
                finally:
                    self.loading_embeddings = False
            
            threading.Thread(target=_load_embeddings, daemon=True).start()
        
        # Kamera başlat (Windows'ta DirectShow backend daha stabil) ve index probe et
        preferred_indices = [0, 1, 2]
        cap = None
        for idx in preferred_indices:
            test = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            if test.isOpened():
                cap = test
                print(f"Camera opened on index {idx}")
                break
            else:
                test.release()
        if cap is None:
            print("The camera did not open.. Tried indices: 0,1,2")
            return False
        
        # Kamera ayarları
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 15)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        try:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        except Exception:
            pass

        # Pencere oluştur
        cv2.namedWindow('Face Recognition', cv2.WINDOW_NORMAL)
        try:
            cv2.resizeWindow('Face Recognition', 800, 600)
        except Exception:
            pass
        
        print(f"TC: {tc_kimlik_input}, Person: {full_name}")
        print("Show your face to the camera...")
        print("Press Q to quit, R to reset. Close the window to exit.")
        
        start_time = time.time()
        no_face_counter = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            self.frame_count += 1
            
            # Pencere kapandıysa çık
            try:
                if cv2.getWindowProperty('Face Recognition', cv2.WND_PROP_VISIBLE) < 1:
                    break
            except Exception:
                pass

            # Frame atlama
            if self.frame_count % self.frame_skip != 0:
                cv2.putText(frame, 'Press Q to quit, R to reset', (10, 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.imshow('Face Recognition', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue
            
            # Hızlı ön işleme
            processed_frame = self.preprocess_frame_fast(frame)
            
            # Kalite değerlendirmesi
            quality = self.assess_frame_quality_fast(processed_frame)
            
            # Embedding'ler hazır değilse bekleme ekranı göster
            if not self.known_embeddings_ready:
                cv2.putText(frame, 'Preparing embeddings...', (10, 150), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(frame, f'Loaded photos: pending', (10, 180), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv2.putText(frame, 'Press Q to quit, R to reset', (10, 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.imshow('Face Recognition', frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), ord('Q'), 27):
                    break
                elif key == ord('r'):
                    self.consecutive_matches = 0
                    print("Reset done.")
                continue

            # Yüz algılama
            faces = self.detect_face_optimized(processed_frame)
            
            status_text = ""
            status_color = (0, 0, 255)  # Kırmızı
            
            if faces:
                no_face_counter = 0
                face_box = faces[0]['box']
                confidence = faces[0]['confidence']
                
                # Yüz çerçevesi çiz
                x, y, w, h = face_box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Embedding çıkar ve karşılaştır
                current_embedding = self.extract_face_embedding_fast(processed_frame, face_box)
                
                if current_embedding is not None and self.known_embeddings:
                    is_match, similarity = self.is_match_with_multiple_embeddings(self.known_embeddings, current_embedding)
                    
                    if is_match:
                        self.consecutive_matches += 1
                        self.consecutive_low_similarity = 0
                        status_text = f"{full_name} - Match {self.consecutive_matches}/{self.required_matches}"
                        status_color = (0, 255, 0)  # Yeşil
                        
                        if self.consecutive_matches >= self.required_matches:
                            cap.release()
                            cv2.destroyAllWindows()
                            # 1) Onay / Reddet
                            ok = self.show_confirmation_fast(full_name)
                            if not ok:
                                return False
                            # 2) İlaç kimin için? Seçim ve gerekirse yetki kontrolü
                            target_tc, target_name = self.prompt_target_patient(tc_kimlik_input, full_name)
                            if target_tc is None:
                                return False
                            if not self.verify_take_permission(tc_kimlik_input, target_tc):
                                messagebox.showerror("Yetki Hatası", f"{full_name} seçilen kişi için ilaç alamaz: {target_name}")
                                return False
                            # 3) Seçilen kişi için atanmış ilaçları göster
                            try:
                                meds = self.fetch_medicines_for_patient(target_tc)
                                self.show_medicines_dialog(target_name or full_name, meds)
                            except Exception as e:
                                print(f"İlaçlar alınırken hata: {e}")
                            return True
                    else:
                        self.consecutive_matches = max(0, self.consecutive_matches - 1)
                        if similarity < self.low_similarity_threshold:
                            self.consecutive_low_similarity += 1
                        else:
                            self.consecutive_low_similarity = 0
                        status_text = f"Face detected - Best Similarity: {similarity:.2f}"
                        status_color = (0, 165, 255)  # Turuncu
                        
                        if self.consecutive_low_similarity >= self.low_similarity_required:
                            print("Low similarity observed repeatedly. Person not recognized.")
                            cap.release()
                            cv2.destroyAllWindows()
                            return False
                
                # Güven seviyesi
                cv2.putText(frame, f"Confidence: {confidence:.2f}", 
                           (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            else:
                no_face_counter += 1
                self.consecutive_matches = max(0, self.consecutive_matches - 1)
                
                if no_face_counter > 20:
                    status_text = "Face not detected - Adjust your position"
                else:
                    status_text = "Searching for a face..."
            
            # Durum bilgileri
            cv2.putText(frame, f"Quality: {quality}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, status_text, (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            cv2.putText(frame, f"Loaded photos: {len(self.known_embeddings) if self.known_embeddings else 0}", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # İlerleme çubuğu
            if self.consecutive_matches > 0:
                progress = int((self.consecutive_matches / self.required_matches) * 300)
                cv2.rectangle(frame, (10, 110), (10 + progress, 130), (0, 255, 0), -1)
                cv2.rectangle(frame, (10, 110), (310, 130), (255, 255, 255), 2)
            
            # FPS göster
            current_time = time.time()
            fps = self.frame_count / (current_time - start_time)
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, frame.shape[0]-20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.putText(frame, 'Press Q to quit, R to reset', (10, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imshow('Face Recognition', frame)
            
            # Çıkış kontrolü
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q'), 27):
                break
            elif key == ord('r'):
                self.consecutive_matches = 0
                print("Reset done.")
        
        cap.release()
        cv2.destroyAllWindows()
        return False
    
    def show_confirmation_fast(self, person_name, target_name=None):
        """Hızlı onay penceresi"""
        print(f"Identity verified: {person_name}")
        
        root = tk.Tk()
        root.title("Verification")
        root.geometry("400x200")
        root.attributes('-topmost', True)
        
        confirmed = tk.BooleanVar()
        confirmed.set(False)
        
        def confirm_yes():
            confirmed.set(True)
            root.quit()
        
        def confirm_no():
            confirmed.set(False)
            root.quit()
        
        header = f"{person_name}\nVerified"
        if target_name:
            header += f"\nİlaç alınacak kişi: {target_name}"
        tk.Label(root, text=header, 
                font=("Arial", 14, "bold")).pack(pady=20)
        
        button_frame = tk.Frame(root)
        button_frame.pack(pady=20)
        
        tk.Button(button_frame, text="✓ CONFIRM", command=confirm_yes,
                 bg='green', fg='white', font=("Arial", 12),
                 width=10, height=2).pack(side=tk.LEFT, padx=10)
        
        tk.Button(button_frame, text="✗ REJECT", command=confirm_no,
                 bg='red', fg='white', font=("Arial", 12),
                 width=10, height=2).pack(side=tk.LEFT, padx=10)
        
        # Otomatik kapanma (10 saniye)
        root.after(10000, lambda: [confirmed.set(True), root.quit()])
        
        # Merkeze hizala
        root.geometry(f"400x200+{root.winfo_screenwidth()//2-200}+{root.winfo_screenheight()//2-100}")
        
        root.mainloop()
        
        result = confirmed.get()
        root.destroy()
        
        if result:
            print("✓ Verification completed.")
        else:
            print("✗ Verification rejected.")
        
        return result

    def fetch_medicines_for_patient(self, patient_tc):
        """Aktif reçetelere bağlı ilaçları döndür."""
        query = (
            """
            SELECT m.name, m.active_ingredient, m.dosage, pm.dosage_instruction
            FROM prescriptions p
            JOIN prescription_medicines pm ON pm.prescription_id = p.id
            JOIN medicines m ON m.id = pm.medicines_id
            WHERE p.patient_tc = %s
              AND p.status = 'active'
              AND (p.start_date IS NULL OR p.start_date <= CURDATE())
              AND (p.end_date IS NULL OR p.end_date >= CURDATE())
            ORDER BY m.name ASC
            """
        )
        self.cursor.execute(query, (patient_tc,))
        rows = self.cursor.fetchall() or []
        meds = []
        for name, active_ing, dosage, instruction in rows:
            meds.append({
                'name': name or '',
                'active_ingredient': active_ing or '',
                'dosage': dosage or '',
                'instruction': instruction or ''
            })
        # dispensing_log geçmişini ekle
        try:
            self.cursor.execute(
                """
                SELECT m.name, dl.quantity, dl.dispensed_at
                FROM dispensing_log dl
                JOIN medicines m ON m.id = dl.medicine_id
                WHERE dl.patient_tc = %s
                ORDER BY dl.dispensed_at DESC
                """,
                (patient_tc,)
            )
            logs = self.cursor.fetchall() or []
            if logs:
                meds.append({'name': '--- Geçmiş ---', 'active_ingredient': '', 'dosage': '', 'instruction': ''})
                for name, qty, ts in logs:
                    meds.append({
                        'name': f"{name} x{qty}",
                        'active_ingredient': '',
                        'dosage': '',
                        'instruction': f"Veriliş: {ts}"
                    })
        except Exception as e:
            print(f"dispensing_log okunamadı: {e}")
        return meds

    def show_medicines_dialog(self, person_name, medicines):
        """Basit bir pencerede ilaçları gösterir."""
        root = tk.Tk()
        root.title("Atanmış İlaçlar")
        root.geometry("520x380")
        root.attributes('-topmost', True)

        tk.Label(root, text=f"{person_name} için atanmış ilaçlar", font=("Arial", 13, "bold")).pack(pady=10)
        frame = tk.Frame(root)
        frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=8)
        txt = tk.Text(frame, wrap=tk.WORD, height=12)
        txt.pack(fill=tk.BOTH, expand=True)
        if not medicines:
            txt.insert(tk.END, "Aktif reçete bulunamadı ya da ilaç tanımlı değil.\n")
        else:
            for i, med in enumerate(medicines, start=1):
                line = f"{i}) {med['name']} - {med['active_ingredient']} {med['dosage']}\n"
                if med['instruction']:
                    line += f"   Talimat: {med['instruction']}\n"
                txt.insert(tk.END, line + "\n")
        tk.Button(root, text="Kapat", command=root.destroy, width=12).pack(pady=8)
        root.update_idletasks()
        root.geometry(f"520x380+{root.winfo_screenwidth()//2-260}+{root.winfo_screenheight()//2-190}")
        root.mainloop()

    def prompt_target_patient(self, requester_tc, requester_name):
        """İlaç alınacak kişiyi seçtir (kendisi veya akrabaları)."""
        try:
            self.cursor.execute(
                """
                SELECT pr.patient_tc AS tc, CONCAT(p.name, ' ', IFNULL(p.surname,'')) AS full_name
                FROM patient_relatives pr
                JOIN patients p ON pr.patient_tc = p.tc_kimlik
                WHERE pr.relative_tc = %s
                UNION ALL
                SELECT p.tc_kimlik AS tc, CONCAT(p.name, ' ', IFNULL(p.surname,'')) AS full_name
                FROM patients p
                WHERE p.tc_kimlik = %s
                """,
                (requester_tc, requester_tc)
            )
            rows = self.cursor.fetchall() or []
        except Exception:
            rows = []

        options = []
        seen = set()
        # Önce kendisi
        try:
            self.cursor.execute("SELECT name, IFNULL(surname,'') FROM patients WHERE tc_kimlik = %s", (requester_tc,))
            self_row = self.cursor.fetchone()
            if self_row:
                self_name = f"{self_row[0]} {self_row[1]}".strip()
            else:
                self_name = requester_name
        except Exception:
            self_name = requester_name
        options.append((requester_tc, f"Kendim - {self_name} ({requester_tc})"))
        seen.add(requester_tc)
        # Akrabaları ekle
        for tc_val, full_name in rows:
            if tc_val in seen:
                continue
            options.append((tc_val, f"{full_name} ({tc_val})"))
            seen.add(tc_val)

        # Seçim UI
        root = tk.Tk()
        root.title("İlaç Kimin İçin?")
        root.geometry("420x300")
        root.attributes('-topmost', True)

        selected_tc = {'value': None}
        selected_name = {'value': None}

        frame = tk.Frame(root)
        frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)

        tk.Label(frame, text="İlaç alınacak kişiyi seçiniz:", font=("Arial", 12, "bold")).pack(pady=(0,10))

        listbox = tk.Listbox(frame, selectmode=tk.SINGLE)
        for idx, (_, label) in enumerate(options):
            listbox.insert(idx, label)
        listbox.pack(fill=tk.BOTH, expand=True)

        def on_ok():
            sel = listbox.curselection()
            if not sel:
                messagebox.showwarning("Seçim Gerekli", "Lütfen bir kişi seçiniz.")
                return
            tc_val, label = options[sel[0]]
            selected_tc['value'] = tc_val
            selected_name['value'] = label.split('(')[0].strip()
            root.quit()

        def on_cancel():
            selected_tc['value'] = None
            selected_name['value'] = None
            root.quit()

        btn_frame = tk.Frame(frame)
        btn_frame.pack(pady=10)
        tk.Button(btn_frame, text="Seç", command=on_ok, bg='green', fg='white', width=10).pack(side=tk.LEFT, padx=8)
        tk.Button(btn_frame, text="Vazgeç", command=on_cancel, bg='red', fg='white', width=10).pack(side=tk.LEFT, padx=8)

        if options:
            listbox.select_set(0)

        root.geometry(f"420x300+{root.winfo_screenwidth()//2-210}+{root.winfo_screenheight()//2-150}")
        root.mainloop()
        root.destroy()

        return selected_tc['value'], selected_name['value']

    def verify_take_permission(self, requester_tc, target_tc):
        """Kişi kendisi veya ilişkili olduğu hasta için ilaç alabilir."""
        if requester_tc == target_tc:
            return True
        try:
            self.cursor.execute(
                """
                SELECT 1
                FROM patient_relatives
                WHERE relative_tc = %s AND patient_tc = %s
                LIMIT 1
                """,
                (requester_tc, target_tc)
            )
            return self.cursor.fetchone() is not None
        except Exception:
            return False
    

    
    def clear_embedding_cache(self):
        """Embedding cache'ini temizle"""
        self.embedding_cache.clear()
        self.face_cache.clear()
        print("Cache cleared.")
    
    def close(self):
        if hasattr(self, 'conn'):
            self.conn.close()

# Ana program
if __name__ == "__main__":
    recognizer = OptimizedFaceRecognizerWithPhotos()
    
    try:
        tc_input = input("TC Kimlik No: ")
        success = recognizer.run_recognition_optimized(tc_input)
        
        if success:
            print("Transaction successful!")
        else:
            print("The transaction could not be completed..")
            
    except KeyboardInterrupt:
        print("\nThe program has been terminated..")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        recognizer.close()