import mysql.connector
import os
import glob
from dotenv import load_dotenv

load_dotenv()

def update_existing_photo_paths():
    conn = mysql.connector.connect(
        host=os.environ.get('DB_HOST', 'localhost'),
        user=os.environ.get('DB_USER', 'root'),
        password=os.environ.get('DB_PASSWORD', ''),
        database=os.environ.get('DB_NAME', 'face_recognition')
    )
    cursor = conn.cursor()
    
    # Ana klasör yolu
    base_path = r"C:\Users\Ezgi\Desktop\Python_repo\face_tanima\known_faces"
    
    # Tüm alt klasörlerdeki jpg dosyalarını bul
    photo_files = glob.glob(os.path.join(base_path, "**", "*.jpg"), recursive=True)
    
    # Patients tablosundan isim-tc eşleştirmesi al
    cursor.execute("SELECT name, surname, tc_kimlik FROM patients")
    patients = cursor.fetchall()
    
    # İsim-TC eşleştirme sözlüğü oluştur
    name_to_tc = {}
    for name, surname, tc in patients:
        name_to_tc[name.lower()] = tc
        # Eğer hem isim hem soyisim kullanan klasörler varsa
        full_name = f"{name} {surname}".lower()
        name_to_tc[full_name] = tc
    
    updated_count = 0
    
    for filepath in photo_files:
        # Klasör adını al (kişi ismi)
        folder_name = os.path.basename(os.path.dirname(filepath)).lower()
        filename = os.path.basename(filepath)
        
        print(f"Processing: {folder_name}/{filename}")
        
        # Klasör adından TC'yi bul
        if folder_name in name_to_tc:
            tc_kimlik = name_to_tc[folder_name]
            
            # Bu TC'ye ait boş photo_path olan kayıtları güncelle
            cursor.execute("""
                UPDATE photos 
                SET photo_path = %s 
                WHERE patient_tc = %s AND photo_path IS NULL
                LIMIT 1
            """, (filepath, tc_kimlik))
            
            if cursor.rowcount > 0:
                updated_count += 1
                print(f"Updated: {tc_kimlik} -> {filepath}")
        else:
            print(f"No patient found for folder: {folder_name}")
    
    conn.commit()
    cursor.close()
    conn.close()
    print(f"Total {updated_count} photo paths updated!")

# Script'i çalıştır
update_existing_photo_paths()