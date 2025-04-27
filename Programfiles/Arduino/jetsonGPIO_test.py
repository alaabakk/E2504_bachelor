import Jetson.GPIO as GPIO
import time

# Use physical board pin numbers
GPIO.setmode(GPIO.BOARD)

# Define PWM pins (must be properly mapped in gpio_pin_data.py)
servo_pins = {
    32: "PWM7",  # Should map to pwmchip3/pwm0
    33: "PWM0"   # Should map to pwmchip0/pwm0
}

# Set up pins and PWM
pwms = {}

for pin in servo_pins:
    GPIO.setup(pin, GPIO.OUT)
    pwm = GPIO.PWM(pin, 50)  # 50 Hz = 20ms period
    pwm.start(7.5)           # Neutral position
    pwms[pin] = pwm

try:
    while True:
        print("Moving to 0°")
        for pwm in pwms.values():
            pwm.ChangeDutyCycle(2.5)
        time.sleep(1)

        print("Moving to 180°")
        for pwm in pwms.values():
            pwm.ChangeDutyCycle(12.5)
        time.sleep(1)

        print("Centering at 90°")
        for pwm in pwms.values():
            pwm.ChangeDutyCycle(7.5)
        time.sleep(1)

except KeyboardInterrupt:
    print("Stopping servos...")

finally:
    for pwm in pwms.values():
        pwm.stop()
    GPIO.cleanup()