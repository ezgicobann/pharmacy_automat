import numpy as np
import mysql.connector
import os
from dotenv import load_dotenv

load_dotenv()

# .npy dosyalarını yükle
embeddings = np.load("embeddings.npy")
labels = np.load('labels.npy')

# Tekrarlanan etiketleri kaldırarak yalnızca benzersiz etiketleri elde edin
unique_labels = np.unique(labels)

# Benzersiz etiketleri yazdırın
print(unique_labels)



# Veritabanı bağlantısı
conn = mysql.connector.connect(
    host=os.environ.get('DB_HOST', 'localhost'),
    user=os.environ.get('DB_USER', 'root'),
    password=os.environ.get('DB_PASSWORD', ''),
    database=os.environ.get('DB_NAME', 'face_recognition')
)
cursor = conn.cursor()

# Her etiket için veritabanında sorgu yap
for label in unique_labels:
    cursor.execute("SELECT id, name, surname, embedding FROM people WHERE name = %s", (label,))
    result = cursor.fetchone()
    
    if result:
        person_id, name, surname, existing_embedding = result
        if existing_embedding is None:  # Sadece null embedding varsa güncelle
            embedding = embeddings[labels == label][0]
            embedding_bytes = embedding.astype(np.float32).tobytes()
            # Veritabanındaki embedding'i güncelle
            cursor.execute("UPDATE people SET embedding = %s WHERE id = %s", (embedding_bytes, person_id))
            print(f"{label} için embedding güncellendi.")
        else:
            print(f"{label} için embedding zaten mevcut, güncellenmiyor.")
    else:
        print(f"{label} veritabanında bulunamadı. Atlanıyor.")

# Değişiklikleri kaydet
conn.commit()
cursor.close()
conn.close()

print("Tüm embeddingler veritabanına yüklendi.")
