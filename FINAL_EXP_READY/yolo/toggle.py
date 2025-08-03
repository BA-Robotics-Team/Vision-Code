import RPi.GPIO as GPIO
import time

# Use BCM pin numbering
GPIO.setmode(GPIO.BCM)

# Set GPIO 14 as output
GPIO_PIN = 14
GPIO_PIN1 = 15

GPIO.setup(GPIO_PIN, GPIO.OUT)
GPIO.setup(GPIO_PIN1, GPIO.OUT)

print(f"Toggling GPIO {GPIO_PIN}... Press Ctrl+C to stop.")

try:
    while True:
        GPIO.output(GPIO_PIN, GPIO.HIGH)  # Set pin HIGH
        time.sleep(2)                     # Wait 1 second
        GPIO.output(GPIO_PIN, GPIO.LOW)   # Set pin LOW
        time.sleep(2)                     # Wait 1 second
        GPIO.output(GPIO_PIN1, GPIO.HIGH)  # Set pin HIGH
        time.sleep(2)                     # Wait 1 second
        GPIO.output(GPIO_PIN1, GPIO.LOW)   # Set pin LOW
        time.sleep(2)              
except KeyboardInterrupt:
    print("Exiting gracefully...")

# Cleanup GPIO settings
finally:
    GPIO.cleanup()
