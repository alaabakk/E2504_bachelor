class ServoPWM:
    def __init__(self, pwmchip, pwmid):
        self.base = f"/sys/class/pwm/pwmchip{pwmchip}"
        self.channel = f"{self.base}/pwm{pwmid}"
        self._export(pwmid)
        self.set_period(20000000)  # 20ms

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
        # 1ms to 2ms pulse width
        duty = 1000000 + (angle / 180.0) * 1000000
        self.set_duty(int(duty))

# Example usage:
servo = ServoPWM(3, 0)  # pwmchip3, pwm0 = pin 32
servo.enable()
servo.set_angle(90)
time.sleep(2)
servo.set_angle(0)
time.sleep(2)
servo.disable()
