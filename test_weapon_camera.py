from ultralytics import YOLO
import cv2
import torch

device = 0 if torch.cuda.is_available() else "cpu"

model = YOLO("models/weapon_best.pt")

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.7, device=device)

    annotated = results[0].plot()

    cv2.imshow("Weapon Detection", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
