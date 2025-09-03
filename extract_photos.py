import mysql.connector
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

def extract_photos_to_files():
    conn = mysql.connector.connect(
        host=os.environ.get('DB_HOST', 'localhost'),
        user=os.environ.get('DB_USER', 'root'),
        password=os.environ.get('DB_PASSWORD', ''),
        database=os.environ.get('DB_NAME', 'face_recognition')
    )
    cursor = conn.cursor()
    
    # Photos klasörünü oluştur
    os.makedirs('backup_photos', exist_ok=True)
    
    # BLOB verilerini al
    cursor.execute("""
        SELECT id, patient_tc, photo 
        FROM photos 
        WHERE photo IS NOT NULL
    """)
    
    for row in cursor.fetchall():
        photo_id, patient_tc, blob_data = row
        
        if blob_data:
            # Dosya adı oluştur
            timestamp = int(datetime.now().timestamp())
            filename = f"patient_{patient_tc}_photo_{photo_id}_{timestamp}.jpg"
            filepath = f"backup_photos/{filename}"
            
            # BLOB'u dosyaya yaz
            with open(filepath, 'wb') as f:
                f.write(blob_data)
            
            # photo_path'i güncelle
            cursor.execute("""
                UPDATE photos 
                SET photo_path = %s 
                WHERE id = %s
            """, (filepath, photo_id))
            
            print(f"Processed: {filename}")
    
    conn.commit()
    cursor.close()
    conn.close()
    print("Photo extraction completed!")

# Script'i çalıştır
extract_photos_to_files()