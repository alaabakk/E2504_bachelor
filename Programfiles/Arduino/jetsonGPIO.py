import Jetson.GPIO as GPIO
import time

# Use physical pin numbers
servo_pin1 = 32  
servo_pin2 = 33  

GPIO.setmode(GPIO.BOARD)
GPIO.setup(servo_pin1, GPIO.OUT)
GPIO.setup(servo_pin2, GPIO.OUT)

# Set up 50Hz PWM (standard for servo control)
pwm1 = GPIO.PWM(servo_pin1, 50)
pwm2 = GPIO.PWM(servo_pin2, 50)

# Start PWM with center (neutral) pulse (7.5% duty cycle)
pwm1.start(7.5)
pwm2.start(7.5)

try:
    while True:
        # Your servo movement logic
        print("Left")
        pwm1.ChangeDutyCycle(5)
        pwm2.ChangeDutyCycle(5)
        time.sleep(1)

        print("Right")
        pwm1.ChangeDutyCycle(10)
        pwm2.ChangeDutyCycle(10)
        time.sleep(1)

        print("Center")
        pwm1.ChangeDutyCycle(7.5)
        pwm2.ChangeDutyCycle(7.5)
        time.sleep(1)

except KeyboardInterrupt:
    print("Interrupted by user")

finally:
    pwm1.stop()
    pwm2.stop()
    try:
        GPIO.cleanup()
    except Exception as e:
        print("GPIO cleanup warning:", e)