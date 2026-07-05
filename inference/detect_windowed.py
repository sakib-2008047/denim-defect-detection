from ultralytics import YOLO
import cv2
import torch

model = YOLO(r"runs\detect\fabric_defect_yolov8n_final\weights\best.pt")
device = 0 if torch.cuda.is_available() else "cpu"
print("Using device:", device)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # change index if needed
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(
        source=frame, conf=0.6, imgsz=640,
        device=device, stream=False, half=False
    )
    annotated = results[0].plot()
    cv2.imshow("Fabric Defect Detection", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
