import os
import time

class ServoPWM:
    def __init__(self, pwmchip, pwmid):
        self.base = f"/sys/class/pwm/pwmchip{pwmchip}"
        self.channel = f"{self.base}/pwm{pwmid}"
        self._export(pwmid)
        self.set_period(20000000)  # 20 ms for 50 Hz

    def _export(self, pwmid):
        if not os.path.exists(self.channel):
            with open(f"{self.base}/export", 'w') as f:
                f.write(str(pwmid))
            time.sleep(0.1)

    def set_period(self, period_ns):
        with open(f"{self.channel}/period", 'w') as f:
            f.write(str(period_ns))

    def set_duty(self, duty_ns):
        with open(f"{self.channel}/duty_cycle", 'w') as f:
            f.write(str(duty_ns))

    def enable(self):
        with open(f"{self.channel}/enable", 'w') as f:
            f.write("1")

    def disable(self):
        with open(f"{self.channel}/enable", 'w') as f:
            f.write("0")

    def set_angle(self, angle):
        # Clamp between 0 and 180
        angle = max(0, min(180, angle))
        # Convert angle to duty cycle (1ms to 2ms pulse)
        duty = 1000000 + (angle / 180.0) * 1000000
        self.set_duty(int(duty))

# ============================
# Setup both servos
# ============================

# Pin 32 = pwmchip3/pwm0
servo1 = ServoPWM(3, 0)

# Pin 33 = pwmchip0/pwm0
servo2 = ServoPWM(0, 0)

servo1.enable()
servo2.enable()

print("Moving both servos...")

try:
    while True:
        print("Position: 0°")
        servo1.set_angle(0)
        servo2.set_angle(0)
        time.sleep(1)

        print("Position: 90°")
        servo1.set_angle(90)
        servo2.set_angle(90)
        time.sleep(1)

        print("Position: 180°")
        servo1.set_angle(180)
        servo2.set_angle(180)
        time.sleep(1)

except KeyboardInterrupt:
    print("Stopping servos...")

finally:
    servo1.disable()
    servo2.disable()
