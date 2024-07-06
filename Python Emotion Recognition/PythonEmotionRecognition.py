import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.optimizers import Adam

face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

model_path = 'fer2013_mini_XCEPTION.110-0.65.hdf5'

if not os.path.exists(model_path):
    raise FileNotFoundError(f'Model dosyası {model_path} bulunamadı.')

emotion_model = load_model(model_path, compile=False)

emotion_model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

emotion_labels = ['Kizgin', 'Tiksinti', 'Korku', 'Mutlu', 'Uzgun', 'Sasirmis', 'Dogal']

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_classifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (64, 64), interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=-1)
            roi = np.expand_dims(roi, axis=0)

            prediction = emotion_model.predict(roi)[0]
            max_index = np.argmax(prediction)
            emotion_label = emotion_labels[max_index]
            emotion_probability = np.max(prediction)

            label_position = (x, y - 10)
            cv2.putText(frame, f'{emotion_label} ({emotion_probability:.2f})', label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, 'Y�z Bulunamadi', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Duygu Analizi', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
