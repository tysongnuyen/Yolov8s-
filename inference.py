import ultralytics
import os
from ultralytics import YOLO
model = YOLO("C:\Dev\ultralytics\gold_ball\train\yolov8n_golf_ball6\weights\best.pt")
if __name__ == "__main__":
    folder = r'C:\Dev\ultralytics\Model_check'
    test_img = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(('.jpg'))]

    for img in test_img:
        results = model.predict(img, save=True, save_txt=True, device='cpu',half=True)
    for img in test_img:
        results = model.predict(img, save=True, save_txt=True, device='cpu', half=False)
    for img in test_img:
        results = model.predict(img, save=True, save_txt=True, device='cuda:0', half=True)
    for img in test_img:
        results = model.predict(img, save=True, save_txt=True, device='cuda:0', half=False)