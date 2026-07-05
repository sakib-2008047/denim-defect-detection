from ultralytics import YOLO
import cv2, torch, socket

# -- Raspberry Pi connection ------------------------------------------------
PI_IP = "192.168.0.101"  
PI_PORT = 65432

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((PI_IP, PI_PORT))
print("Connected to Raspberry Pi")

# -- Model and camera setup --------------------------------------------------
model = YOLO(r"runs\detect\fabric_defect_yolov8n_final\weights\best.pt")
device = 0 if torch.cuda.is_available() else "cpu"

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Full-screen display
cv2.namedWindow("Fabric Defect Detection", cv2.WINDOW_NORMAL)
cv2.setWindowProperty(
    "Fabric Defect Detection",
    cv2.WND_PROP_FULLSCREEN,
    cv2.WINDOW_FULLSCREEN
)

last_state = None

# -- Main detection loop -----------------------------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(
        source=frame, conf=0.6, imgsz=640,
        device=device, stream=False, half=False
    )
    boxes = results[0].boxes

    state = "DEFECT" if (boxes is not None and len(boxes) > 0) else "OK"

    # Transmit only on state change
    if state != last_state:
        sock.sendall(state.encode())
        print("Sent to Pi:", state)
        last_state = state

    annotated = results[0].plot()
    cv2.imshow("Fabric Defect Detection", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# -- Cleanup ------------------------------------------------------------------
sock.close()
cap.release()
cv2.destroyAllWindows()
