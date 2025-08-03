import serial
import time

try:
    ser = serial.Serial('/dev/ttyAMA0', 9600, timeout=1)
    print("Serial port opened successfully.")
except Exception as e:
    print("Failed to open serial port:", e)
    exit()

count = 0
while True:
    msg = f"Hello {count} from Raspberry Pi\r\n"
    ser.write(msg.encode())
    print(f"Sent: {msg.strip()}")
    count += 1
    time.sleep(1)
    
