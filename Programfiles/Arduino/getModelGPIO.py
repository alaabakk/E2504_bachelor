import Jetson.GPIO as GPIO

print("Model:", GPIO.model)

GPIO.setmode(GPIO.BOARD)
GPIO.setup(32, GPIO.OUT)
pwm = GPIO.PWM(32, 50)
pwm.start(7.5)