import serial
import time

ser = serial.Serial('/dev/ttyAMA0', 9600, timeout=1)
print("Serial port opened")

flag1=0
count=0
while True:
	msg = f"Hello {count} from Raspberry Pi\r\n"
	ser.write(msg.encode())
	print(msg)
	time.sleep(0.2)
	#flag1=0
	line = ser.readline().decode(errors='ignore').strip()

	if line.strip() == "Hello":
			print("Received:", line)
			#flag1=1
			break
	else:
		print(f"Wrong Acknowledgement Received {line}\r\n")
	count+=1
    
