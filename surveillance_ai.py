from ultralytics import YOLO
import cv2
import torch
import datetime
import os
import time
import math
from deep_sort_realtime.deepsort_tracker import DeepSort


device = 0 if torch.cuda.is_available() else "cpu"
print("Running on:", "GPU" if device == 0 else "CPU")



general_model = YOLO("yolov8x.pt")
weapon_model = YOLO("runs/detect/train/weights/best.pt")



tracker = DeepSort(max_age=30)
previous_positions = {}



cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH,1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,1080)



os.makedirs("evidence", exist_ok=True)



alert_log = []
last_alert_time = 0
alert_cooldown = 120


while True:

    ret, frame = cap.read()

    if not ret:
        break

    annotated = frame.copy()

    person_count = 0
    weapon_count = 0
    weapon_type = None

    detections = []

  
    results = general_model(frame, conf=0.5, imgsz=960, device=device)

    for box in results[0].boxes:

        cls_id = int(box.cls[0])
        label = general_model.names[cls_id]
        conf = float(box.conf[0])

        x1,y1,x2,y2 = map(int, box.xyxy[0])

        cv2.rectangle(annotated,(x1,y1),(x2,y2),(0,255,0),2)

        text = f"{label} {conf:.2f}"

        cv2.putText(
            annotated,
            text,
            (x1,y1-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0,255,0),
            2
        )

        if label == "person":

            person_count += 1

            detections.append(([x1,y1,x2-x1,y2-y1], conf, label))

        if label == "knife":

            weapon_count += 1
            weapon_type = "Knife"

   

    tracks = tracker.update_tracks(detections, frame=frame)

    suspicious_activity = False

    for track in tracks:

        if not track.is_confirmed():
            continue

        track_id = track.track_id

        l,t,r,b = map(int, track.to_ltrb())

        cx = int((l+r)/2)
        cy = int((t+b)/2)

        if track_id in previous_positions:

            px,py = previous_positions[track_id]

            distance = math.sqrt((cx-px)**2 + (cy-py)**2)

            if distance > 80:

                suspicious_activity = True

        previous_positions[track_id] = (cx,cy)

        cv2.rectangle(annotated,(l,t),(r,b),(255,255,0),2)

        cv2.putText(
            annotated,
            f"Person ID {track_id}",
            (l,t-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255,255,0),
            2
        )

   

    weapon_results = weapon_model(frame, conf=0.45, imgsz=960, device=device)

    for box in weapon_results[0].boxes:

        cls_id = int(box.cls[0])
        label = weapon_model.names[cls_id]
        conf = float(box.conf[0])

        weapon_count += 1
        weapon_type = label

        x1,y1,x2,y2 = map(int, box.xyxy[0])

        cv2.rectangle(annotated,(x1,y1),(x2,y2),(0,0,255),3)

        text = f"{label} {conf:.2f}"

        cv2.putText(
            annotated,
            text,
            (x1,y1-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0,0,255),
            2
        )

    

    threat = "LOW"

    if suspicious_activity:
        threat = "MEDIUM"

    if weapon_count > 0 and person_count > 0:
        threat = "HIGH"

    elif weapon_count > 0:
        threat = "MEDIUM"

  
    color = (0,255,0)

    if threat == "MEDIUM":
        color = (0,255,255)

    if threat == "HIGH":
        color = (0,0,255)

    cv2.putText(
        annotated,
        f"Threat Level: {threat}",
        (30,50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        color,
        3
    )

    

    if suspicious_activity:

        cv2.putText(
            annotated,
            "⚠ Suspicious Activity Detected",
            (30,90),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,0,255),
            3
        )

  

    current_time = time.time()

    if threat == "HIGH" and weapon_type is not None and (current_time - last_alert_time) > alert_cooldown:

        timestamp = datetime.datetime.now().strftime("%H:%M:%S")

        filename = f"evidence/threat_{timestamp.replace(':','')}.jpg"

        cv2.imwrite(filename, annotated)

        log_entry = f"{timestamp}  HIGH THREAT  {weapon_type} detected"

        alert_log.append(log_entry)

        print(log_entry)

        last_alert_time = current_time


    y_offset = 130

    cv2.putText(
        annotated,
        "ALERT LOG",
        (30,y_offset),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255,255,255),
        2
    )

    for entry in alert_log[-5:]:

        y_offset += 30

        cv2.putText(
            annotated,
            entry,
            (30,y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255,255,255),
            1
        )

   
    cv2.namedWindow("AI Defense Surveillance", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("AI Defense Surveillance", 900, 720)
    cv2.imshow("AI Defense Surveillance", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break



cap.release()
cv2.destroyAllWindows()