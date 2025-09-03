import mysql.connector
import os
from dotenv import load_dotenv

load_dotenv()

def check_photos():
    conn = mysql.connector.connect(
        host=os.environ.get('DB_HOST', 'localhost'),
        user=os.environ.get('DB_USER', 'root'),
        password=os.environ.get('DB_PASSWORD', ''),
        database=os.environ.get('DB_NAME', 'face_recognition')
    )
    cursor = conn.cursor()
    
    # Fotoğraf bilgilerini kontrol et
    cursor.execute("""
        SELECT id, patient_tc, 
               CASE WHEN embedding IS NOT NULL THEN CHAR_LENGTH(embedding) ELSE 0 END as file_size
        FROM photos 
        ORDER BY id
    """)
    
    photos = cursor.fetchall()
    
    print("Photos Tablosu İçeriği:")
    print("-" * 40)
    for id, patient_tc, file_size in photos:
        status = "Var" if file_size > 0 else "Yok"
        print(f"ID: {id} | TC: {patient_tc} | Dosya: {status} ({file_size} bytes)")
    
    cursor.close()
    conn.close()

# Çalıştır
check_photos()