import cv2
import numpy as np
from mtcnn import MTCNN
from keras_facenet import FaceNet
import mysql.connector
import time
import tkinter as tk
from tkinter import messagebox
from tkinter import simpledialog
from sklearn.metrics.pairwise import cosine_similarity
import warnings
import os
from dotenv import load_dotenv
warnings.filterwarnings('ignore')

load_dotenv()

class OptimizedFaceRecognizer:
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
        
        # Performans için cache
        self.face_cache = []
        self.cache_size = 5
        
        # Validation parametreleri
        self.required_matches = 2  # 3'ten 2'ye düşürüldü
        self.consecutive_matches = 0
        
        # Frame atlama
        self.frame_skip = 3  # Her 3 frame'de bir işle
        self.frame_count = 0
        
        # Yüz algılama optimizasyonu
        self.last_face_box = None
        self.face_lost_frames = 0
        
    def preprocess_frame_fast(self, frame):
        """Hızlı görüntü ön işleme"""
        # Sadece gerekli durumlarda iyileştirme uygula
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        
        if brightness < 80:  # Çok karanlık ise
            # Basit gamma düzeltmesi
            gamma = 1.5
            table = np.array([((i / 255.0) ** (1.0/gamma)) * 255
                             for i in np.arange(0, 256)]).astype("uint8")
            frame = cv2.LUT(frame, table)
        
        return frame
    
    def detect_face_optimized(self, frame):
        """Optimize edilmiş yüz algılama"""
        # ROI kullanarak algılama alanını sınırla
        h, w = frame.shape[:2]
        
        if self.last_face_box is not None:
            # Son yüz pozisyonu etrafında arama yap
            x, y, fw, fh = self.last_face_box
            
            # ROI genişlet (%30 margin)
            margin = 0.3
            roi_x = max(0, int(x - fw * margin))
            roi_y = max(0, int(y - fh * margin))
            roi_w = min(w - roi_x, int(fw * (1 + 2*margin)))
            roi_h = min(h - roi_y, int(fh * (1 + 2*margin)))
            
            roi = frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
            
            try:
                results = self.detector.detect_faces(roi)
                if results and results[0]['confidence'] > 0.8:
                    # ROI koordinatlarını tam frame'e dönüştür
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
        
        # ROI'de bulunamazsa veya ilk algılama ise tam frame'de ara
        if self.last_face_box is None or self.face_lost_frames > 5:
            try:
                # Görüntüyü küçültmüş olarak işle
                scale = 0.5
                small_frame = cv2.resize(frame, (int(w*scale), int(h*scale)))
                
                results = self.detector.detect_faces(small_frame)
                if results:
                    # En güvenilir yüzü al
                    best_result = max(results, key=lambda x: x['confidence'])
                    if best_result['confidence'] > 0.7:
                        # Koordinatları orijinal boyuta dönüştür
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
        
        # Güvenli sınırlar
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
                # Basit benzerlik kontrolü (MSE)
                if np.mean((cached_face - face) ** 2) < 1000:
                    return cached_embedding
            
            # Yeni embedding hesapla
            embedding = self.embedder.embeddings([face])[0]
            
            # Cache'e ekle
            if len(self.face_cache) >= self.cache_size:
                self.face_cache.pop(0)
            self.face_cache.append((face, embedding))
            
            return embedding
            
        except Exception as e:
            print(f"Embedding error: {e}")
            return None
    
    def is_match_fast(self, known_embedding, current_embedding):
        """Hızlı eşleşme kontrolü"""
        # Sadece cosine similarity kullan (daha hızlı)
        similarity = cosine_similarity([known_embedding], [current_embedding])[0][0]
        return similarity > (1 - self.base_threshold), similarity
    
    def assess_frame_quality_fast(self, frame):
        """Hızlı kalite değerlendirmesi"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        
        # Sadece aydınlık kontrolü (hız için)
        if brightness > 100:
            return 'good'
        elif brightness > 60:
            return 'medium'
        else:
            return 'poor'
    
    def run_recognition_optimized(self, tc_kimlik_input):
        """Optimize edilmiş ana tanıma işlemi"""
        # Veritabanından kişi bilgilerini al
        self.cursor.execute("SELECT id, name, embedding FROM patients WHERE tc_kimlik = %s", (tc_kimlik_input,))
        result = self.cursor.fetchone()
        
        if not result:
            print("No person with this ID number was found..")
            return False
        
        person_id, person_name, embedding_blob = result
        known_embedding = np.frombuffer(embedding_blob, dtype=np.float32)
        
        # Kamera başlat
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("The camera did not open..")
            return False
        
        # Kamera ayarları - performans için optimize et
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 15)  # FPS düşür
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Buffer azalt
        
        print(f"TC: {tc_kimlik_input}, Kişi: {person_name}")
        print("Show your face to the camera...")
        
        start_time = time.time()
        no_face_counter = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            self.frame_count += 1
            
            # Frame atlama
            if self.frame_count % self.frame_skip != 0:
                cv2.imshow('Face Recognition', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue
            
            # Hızlı ön işleme
            processed_frame = self.preprocess_frame_fast(frame)
            
            # Kalite değerlendirmesi
            quality = self.assess_frame_quality_fast(processed_frame)
            
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
                
                if current_embedding is not None:
                    is_match, similarity = self.is_match_fast(known_embedding, current_embedding)
                    
                    if is_match:
                        self.consecutive_matches += 1
                        status_text = f"{person_name} - Match {self.consecutive_matches}/{self.required_matches}"
                        status_color = (0, 255, 0)  # Yeşil
                        
                        if self.consecutive_matches >= self.required_matches:
                            cap.release()
                            cv2.destroyAllWindows()
                            # İlaç kimin için alınacak seçim ekranı ve ilişki kontrolü
                            target_tc, target_name = self.prompt_target_patient(tc_kimlik_input, person_name)
                            if target_tc is None:
                                return False
                            if not self.verify_take_permission(tc_kimlik_input, target_tc):
                                messagebox.showerror("Yetki Hatası", f"{person_name} seçilen kişi için ilaç alamaz: {target_name}")
                                return False
                            return self.show_confirmation_fast(person_name, target_name)
                    else:
                        self.consecutive_matches = max(0, self.consecutive_matches - 1)
                        status_text = f"Face detected - Similarity: {similarity:.2f}"
                        status_color = (0, 165, 255)  # Turuncu
                
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
            
            # İlerleme çubuğu
            if self.consecutive_matches > 0:
                progress = int((self.consecutive_matches / self.required_matches) * 300)
                cv2.rectangle(frame, (10, 80), (10 + progress, 100), (0, 255, 0), -1)
                cv2.rectangle(frame, (10, 80), (310, 100), (255, 255, 255), 2)
            
            # FPS göster
            current_time = time.time()
            fps = self.frame_count / (current_time - start_time)
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, frame.shape[0]-20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow('Face Recognition', frame)
            
            # Çıkış kontrolü
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
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
        root.attributes('-topmost', True)  # En üstte göster
        
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
        
        tk.Button(button_frame, text="✓ APPROVE", command=confirm_yes,
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

    def prompt_target_patient(self, requester_tc, requester_name):
        """İlaç alınacak kişiyi seçtir (kendisi veya akrabaları)."""
        try:
            # Kişinin akrabalarını getir
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
            # Tablo yoksa veya hata varsa sadece kendisi
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

        # Basit seçim penceresi
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
            # Extract name part before (tc)
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

        # Varsayılan olarak ilkini seç
        if options:
            listbox.select_set(0)

        # Merkezle
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
            # Tablo yoksa güvenlik için izin verme
            return False
    
    def close(self):
        if hasattr(self, 'conn'):
            self.conn.close()

# Ana program
if __name__ == "__main__":
    recognizer = OptimizedFaceRecognizer()
    
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