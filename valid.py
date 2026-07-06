from ultralytics import YOLO

model = YOLO(r"C:\Dev\ultralytics\gold_ball\train\yolov8n_golf_ball6\weights\best.pt")
if __name__ == "__main__":
    metrics = model.val(data=r"C:\Dev\ultralytics\data.yaml", imgsz=480, batch=12, device="cuda:0", workers=8)
    metrics.box.map50
    metrics.box.map75
    metrics.box.map
    print(f"Validation metrics: {metrics}")
    print(f"Precision: {metrics.box.p}")
