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

# Veritabanından fotoğrafı al
def get_photo_from_db(person_id):
    try:
        cursor.execute("SELECT photo FROM photos WHERE person_id = %s LIMIT 1", (person_id,))
        result = cursor.fetchone()

        if result:
            photo_bytes = result[0]
            nparr = np.frombuffer(photo_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # Fotoğrafı yeniden boyutlandır
            height, width = img.shape[:2]
            aspect_ratio = width / height
            new_width = 400  # Yeni genişlik
            new_height = int(new_width / aspect_ratio)  # Oranlı yükseklik

            # Boyutlandırmayı uygula
            img_resized = cv2.resize(img, (new_width, new_height))

            # Fotoğrafı göster
            cv2.imshow('Fotograf', img_resized)
            cv2.waitKey(0)  # Kullanıcı bir tuşa basana kadar bekle
            cv2.destroyAllWindows()
        else:
            print("Fotoğraf bulunamadı.")
    except mysql.connector.Error as err:
        print(f"Veritabanı hatası: {err}")
    except Exception as e:
        print(f"Bir hata oluştu: {e}")

# Kişi ID'si
person_id = 4

# Fotoğrafı veritabanından al ve göster
get_photo_from_db(person_id)

# Bağlantıyı ve cursor'ı kapat
cursor.close()
conn.close()
