from ultralytics import YOLO

def train():

    model = YOLO("models/yolov8s.pt")

    model.train(
        data="weapon_dataset/data.yaml",
        epochs=50,
        imgsz=640,
        batch=4,
        workers=2,
        device=0
    )

if __name__ == "__main__":
    train()
