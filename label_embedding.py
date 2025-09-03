import os
import cv2
import numpy as np
from keras_facenet import FaceNet
from mtcnn import MTCNN

detector = MTCNN()
embedder = FaceNet()

base_dir = "known_faces"
all_embeddings = []
all_labels = []

for person_name in os.listdir(base_dir):
    person_dir = os.path.join(base_dir, person_name)
    if not os.path.isdir(person_dir):
        continue
    for filename in os.listdir(person_dir):
        img_path = os.path.join(person_dir, filename)
        img = cv2.imread(img_path)
        results = detector.detect_faces(img)
        if not results:
            continue
        x, y, w, h = results[0]['box']
        face = img[y:y+h, x:x+w]
        face = cv2.resize(face, (160, 160))
        embedding = embedder.embeddings([face])[0]
        all_embeddings.append(embedding)
        all_labels.append(person_name)

# Dosyaları kaydet
np.save("embeddings.npy", np.array(all_embeddings))
np.save("labels.npy", np.array(all_labels))
print("embedding ve label dosyaları oluşturuldu.")
