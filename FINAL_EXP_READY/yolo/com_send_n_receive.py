import serial
import time
import threading

# Initialize serial port
ser = serial.Serial('/dev/ttyAMA0', 9600, timeout=1)

def receive_data():
    while True:
        if ser.in_waiting > 0:
            line = ser.readline().decode(errors='ignore').strip()
            if line:
                print("Received:", line)

def send_data():
    while True:
        message = "Hello from Raspberry Pi!\r\n"
        ser.write(message.encode())
        print("Sent:", message.strip())
        time.sleep(3)

try:
    # Start receiver in background
    receiver_thread = threading.Thread(target=receive_data, daemon=True)
    receiver_thread.start()

    # Start sending loop in main thread
    send_data()

except KeyboardInterrupt:
    print("\nExiting.")
    ser.close()
