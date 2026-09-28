from ultralytics import YOLO

model = YOLO("yolov8s.pt")
model.export(imgsz=360, opset=12)
if __name__ == "__main__":
    model.train(
        data="data.yaml",
        epochs=2,
        imgsz=480,
        batch=12,
        device="cuda:0",
        workers=8,
        project="gold_ball/train",
        name="yolov8n_golf_ball",
        save=True,
    )
