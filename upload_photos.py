import mysql.connector
import cv2
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()

# Veritabanı bağlantısı
conn = mysql.connector.connect(
    host=os.environ.get('DB_HOST', 'localhost'),
    user=os.environ.get('DB_USER', 'root'),
    password=os.environ.get('DB_PASSWORD', ''),
    database=os.environ.get('DB_NAME', 'face_recognition')
)
cursor = conn.cursor()

# Fotoğrafı oku ve veritabanına ekle
def save_photo_to_db(photo_path, person_id):
    # Fotoğrafı oku
    img = cv2.imread(photo_path)
    
    # Fotoğrafı JPEG formatında encode et ve byte dizisi olarak al
    _, photo_bytes = cv2.imencode('.jpg', img)
    photo_bytes = photo_bytes.tobytes()  # Byte dizisine çevir
    
    # Veritabanına ekleme işlemi
    cursor.execute(""" 
        INSERT INTO photos (person_id, photo) 
        VALUES (%s, %s)
    """, (person_id, photo_bytes))
    conn.commit()

# Fotoğrafları veritabanına ekle
def add_photos_to_db_for_person(photo_paths, person_id):
    for photo_path in photo_paths:
        save_photo_to_db(photo_path, person_id)

# Kişiler ve klasörleri
people = {
    "Ezgi": 1,
    "Enes": 2,
    "Aylin": 5,
    "Gulsen": 3,
    "Mahmut": 4
}

# Klasördeki her kişi için fotoğrafları veritabanına ekle
base_folder_path = r"C:\Users\Ezgi\Desktop\Python_repo\face_tanima\known_faces"

for person_name, person_id in people.items():
    # Her kişi için fotoğrafların bulunduğu klasörü belirle
    folder_path = os.path.join(base_folder_path, person_name)
    
    if os.path.exists(folder_path):
        # Klasördeki tüm .jpg dosyalarını al
        photo_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.jpg') or f.endswith('.png')]
        
        # Fotoğrafları veritabanına ekle
        add_photos_to_db_for_person(photo_paths, person_id)
        print(f"{person_name} için fotoğraflar başarıyla eklendi.")
    else:
        print(f"{person_name} için klasör bulunamadı: {folder_path}")

# Bağlantıyı kapat
conn.close()
