from ultralytics import YOLO
import cv2, socket

model = YOLO(r"runs\detect\fabric_defect_yolov8n_final\weights\best.pt")
cap = cv2.VideoCapture(0)

# Full-screen window
cv2.namedWindow("Fabric Defect Detection", cv2.WINDOW_NORMAL)
cv2.setWindowProperty(
    "Fabric Defect Detection",
    cv2.WND_PROP_FULLSCREEN,
    cv2.WINDOW_FULLSCREEN
)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.6, imgsz=640, stream=False)
    annotated = results[0].plot()
    cv2.imshow("Fabric Defect Detection", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
