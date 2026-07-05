import csv
import statistics as st

T_INFERENCE_MS = 9.86   # <-- from your paper (end-to-end per-frame, Table 4). Edit if needed.

def load(path, col):
    vals = []
    try:
        with open(path) as f:
            for row in csv.DictReader(f):
                vals.append(float(row[col]))
    except FileNotFoundError:
        print(f"!! missing file: {path}")
    return vals

rtt = load("pc_rtt_ms.csv", "rtt_ms")
act = load("pi_actuation_ms.csv", "actuation_ms")

def ms(x):
    return (st.mean(x), st.pstdev(x), len(x)) if x else (float("nan"), float("nan"), 0)

comm = [r/2 for r in rtt]                 # one-way communication
comm_m, comm_sd, comm_n = ms(comm)
act_m,  act_sd,  act_n  = ms(act)

loop_m = T_INFERENCE_MS + comm_m + act_m
# SD of a sum of independent terms; inference treated as fixed
loop_sd = (comm_sd**2 + act_sd**2) ** 0.5

print("\n================  CLOSED-LOOP LATENCY  ================")
print(f"  T_inference   : {T_INFERENCE_MS:7.3f} ms        (fixed, from Table 4)")
print(f"  T_comm        : {comm_m:7.3f} +/- {comm_sd:5.3f} ms   (n={comm_n})")
print(f"  T_actuation   : {act_m:7.3f} +/- {act_sd:5.3f} ms   (n={act_n})")
print(f"  ---------------------------------------------")
print(f"  T_loop (total): {loop_m:7.3f} +/- {loop_sd:5.3f} ms")
print("======================================================\n")

print("Manuscript table (paste into Results):\n")
print("  Component                 Mean +/- SD (ms)")
print("  -----------------------------------------")
print(f"  Inference (T_inf)         {T_INFERENCE_MS:.2f}")
print(f"  Communication (T_comm)    {comm_m:.2f} +/- {comm_sd:.2f}")
print(f"  Actuation (T_act)         {act_m:.2f} +/- {act_sd:.2f}")
print(f"  End-to-end loop (T_loop)  {loop_m:.2f} +/- {loop_sd:.2f}")
print()
# Context: distance of fabric travelled during the loop, at a given speed
for v_cm_s in (4.58, 33.0):  # your max lab speed, and documented production speed
    dist_mm = loop_m/1000.0 * v_cm_s * 10.0
    print(f"  At {v_cm_s:5.2f} cm/s, fabric travels {dist_mm:5.2f} mm during T_loop")
