import socket
import time
import csv
from gpiozero import LED      # using the same library your real system uses

PIN = 17                      # any free GPIO pin; nothing needs to be attached
pin = LED(PIN)                # 'LED' here just means a digital output pin

HOST, PORT = "192.168.0.101", 65432
actuation_ms = []

def set_high_and_time():
    """Time from here to the GPIO line being driven HIGH = T_act."""
    t0 = time.perf_counter()
    pin.on()                  # <-- the GPIO state change being timed
    t1 = time.perf_counter()
    return (t1 - t0) * 1000.0

print(f"[pi] GPIO{PIN} ready (nothing needs to be connected).")
print(f"[pi] listening on {HOST}:{PORT} ...  Ctrl-C to stop.")

srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
srv.bind((HOST, PORT)); srv.listen(1)

try:
    conn, addr = srv.accept()
    print(f"[pi] connected by {addr}")
    with conn:
        buf = b""
        while True:
            data = conn.recv(64)
            if not data:
                break
            buf += data
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                msg = line.decode().strip().upper()
                if msg == "PING":
                    conn.sendall(b"PONG\n")
                elif msg == "DEFECT":
                    actuation_ms.append(set_high_and_time())
                    conn.sendall(b"ACK\n")
                elif msg == "OK":
                    pin.off()
                elif msg == "QUIT":
                    raise KeyboardInterrupt
except KeyboardInterrupt:
    pass
finally:
    pin.off()
    with open("pi_actuation_ms.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["actuation_ms"])
        for v in actuation_ms:
            w.writerow([f"{v:.4f}"])
    n = len(actuation_ms)
    print(f"[pi] saved {n} actuation samples -> pi_actuation_ms.csv")
    if n:
        print(f"[pi] T_act mean = {sum(actuation_ms)/n:.4f} ms over {n} trials")
    srv.close()
