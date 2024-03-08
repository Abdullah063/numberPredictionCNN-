import cv2
import numpy as np
from keras.models import load_model

# Modeli yükle
model = load_model("model_trained_new.h5")

def preProcess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img / 255
    return img

cap = cv2.VideoCapture(0)
cap.set(3, 480)
cap.set(4, 480)

while True:
    success, frame = cap.read()
    
    img = cv2.resize(frame, (32, 32))
    img_gray = preProcess(img)
    img_input = img_gray.reshape(-1, 32, 32, 1)
    
    # Tahmin yap
    predictions = model.predict(img_input)
    classIndex = np.argmax(predictions, axis=-1)
    probVal = np.amax(predictions) * 100  # Olasılığı yüzde cinsine çevir
    probVal = round(probVal, 2)  # Yuvarla
    
    if probVal > 70:  # Yüzde 70 ve üstü olasılıkları kabul edelim
        cv2.putText(frame, f"Class: {classIndex}   Probability: {probVal}%", (50, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 1)

    cv2.imshow("Rakam Sınıflandırma", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
