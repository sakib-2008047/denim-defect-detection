import socket
from gpiozero import LED, Buzzer, Motor

HOST = "192.168.0.101"
PORT = 65432

# -- Peripheral setup (VERIFY against your actual wiring) --------------------
motor = Motor(forward=23, backward=24)   # L298N IN1 / IN2
green_led = LED(27)                      # pass / OK indicator
red_led = LED(17)                        # defect indicator
buzzer = Buzzer(22)                      # audible alert


def set_ok_state():
    motor.forward()      # keep fabric transport running
    green_led.on()
    red_led.off()
    buzzer.off()


def set_defect_state():
    motor.stop()          # immediate stop
    green_led.off()
    red_led.on()
    buzzer.on()


# -- TCP server loop -----------------------------------------------------------
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind((HOST, PORT))
server.listen(1)
print(f"Listening on {HOST}:{PORT} ...")

conn, addr = server.accept()
print("Connected by", addr)

set_ok_state()  # start in the nominal state

try:
    while True:
        data = conn.recv(1024)
        if not data:
            break

        state = data.decode().strip()

        if state == "DEFECT":
            set_defect_state()
            print("Received: DEFECT -> motor stopped, red LED + buzzer on")
        elif state == "OK":
            set_ok_state()
            print("Received: OK -> motor running, green LED on")
        else:
            print("Unrecognised signal:", state)

finally:
    conn.close()
    server.close()
    motor.stop()
    green_led.off()
    red_led.off()
    buzzer.off()
