import socket
import time
import csv
import statistics as st

# ---- EDIT THESE ----------------------------------------------------------
PI_IP   = "172.20.10.3"   # <-- your Raspberry Pi's IP (same as in your detection script)
PI_PORT = 65432
TRIALS  = 30              # 20-30 recommended; repeat at each fabric speed if you like
GAP_S   = 0.1            # small pause between trials
# --------------------------------------------------------------------------

def recv_line(sock):
    buf = b""
    while b"\n" not in buf:
        chunk = sock.recv(64)
        if not chunk:
            break
        buf += chunk
    return buf.decode().strip()

print(f"[pc] connecting to {PI_IP}:{PI_PORT} ...")
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((PI_IP, PI_PORT))
s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)  # don't buffer tiny packets
print("[pc] connected. Measuring network round-trip (T_comm)...")

rtt_ms = []
for i in range(TRIALS):
    t0 = time.perf_counter()
    s.sendall(b"PING\n")
    reply = recv_line(s)            # expect "PONG"
    t1 = time.perf_counter()
    if reply == "PONG":
        rtt = (t1 - t0) * 1000.0
        rtt_ms.append(rtt)
        print(f"  trial {i+1:2d}: RTT = {rtt:6.3f} ms")
    time.sleep(GAP_S)

with open("pc_rtt_ms.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["rtt_ms"])
    for v in rtt_ms:
        w.writerow([f"{v:.4f}"])

if rtt_ms:
    mean = st.mean(rtt_ms); sd = st.pstdev(rtt_ms)
    print("\n[pc] ===== RESULTS =====")
    print(f"[pc] network RTT : {mean:.3f} +/- {sd:.3f} ms  (n={len(rtt_ms)})")
    print(f"[pc] T_comm (one-way = RTT/2): {mean/2:.3f} +/- {sd/2:.3f} ms")
    print("[pc] saved pc_rtt_ms.csv")
    print("[pc] Now copy pi_actuation_ms.csv off the Pi and run analyze_latency.py")

# Tell the Pi to shut down cleanly (optional). Comment out to keep it running.
# s.sendall(b"QUIT\n")
s.close()
