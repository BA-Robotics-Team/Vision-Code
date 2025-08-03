import serial

ser = serial.Serial('/dev/ttyAMA0', 9600, timeout=1)

print("Listening on /dev/ttyAMA0... Press Ctrl+C to exit.")

try:
    while True:
        if ser.in_waiting > 0:
            line = ser.readline().decode(errors='ignore').strip()
            print("Received:", line)
except KeyboardInterrupt:
    print("Exiting.")
    ser.close()
