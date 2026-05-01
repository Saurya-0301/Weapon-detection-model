from ultralytics import YOLO
import cv2
import torch

device = 0 if torch.cuda.is_available() else "cpu"
print("Running on:", "GPU" if device == 0 else "CPU")

model = YOLO("yolov8m.pt")

cap = cv2.VideoCapture(0)

cap.set(3, 640)
cap.set(4, 480)

# weapon classes
WEAPON_CLASSES = ["knife", "scissors", "baseball bat"]

while True:

    ret, frame = cap.read()

    if not ret:
        break

    results = model(frame, conf=0.5, device=device)

    annotated = results[0].plot()

    threat_level = "LOW"

    if results[0].boxes is not None:

        for box in results[0].boxes:

            class_id = int(box.cls[0])
            class_name = model.names[class_id]

            if class_name in WEAPON_CLASSES:
                threat_level = "HIGH"

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,255), 3)

                cv2.putText(frame,
                            "WEAPON DETECTED",
                            (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0,0,255),
                            2)

    # threat display
    cv2.putText(frame,
                f"Threat Level: {threat_level}",
                (20,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,0,255) if threat_level=="HIGH" else (0,255,0),
                2)

    cv2.imshow("AI Defense Surveillance", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()