import Jetson.GPIO as GPIO
import time

print("Model:", GPIO.model)

GPIO.setmode(GPIO.BOARD)
GPIO.setup(32, GPIO.OUT)

# Create PWM object and print internal state
pwm = GPIO.PWM(32, 50)
print(f"PWM object created: {pwm}")
print(f"PWM dir: {pwm._pwm_dir}")  # This shows /sys/class/pwm/pwmchipX
print(f"PWM chip/channel: {pwm.chip}, {pwm.channel}")

pwm.start(7.5)
time.sleep(2)
pwm.stop()
GPIO.cleanup()