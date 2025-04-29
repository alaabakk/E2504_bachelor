#include <Arduino.h>
#include <ESP32Servo.h>

// -------------------- Pins --------------------
static const int servoPanPin = 18;
static const int servoTiltPin = 19;

#define ENC_PAN_A 32
#define ENC_PAN_B 33

#define ENC_TILT_A 34
#define ENC_TILT_B 35

#define PPR 4096

// -------------------- Globals --------------------
Servo servoPan;
Servo servoTilt;

volatile long encoderCountPan = 0;
volatile long encoderCountTilt = 0;

volatile uint8_t lastEncodedPan = 0;
volatile uint8_t lastEncodedTilt = 0;

float maxSpeed = 90.0; // grader per sekund ved 100% pådrag
float servoPanPos = 90.0;
float servoTiltPos = 56.0;

float SP_pan;
float SP_tilt;

unsigned long lastTime = 0;

// -------------------- Encoder Interrupts --------------------
void IRAM_ATTR updateEncoderPan() {
  bool MSB = digitalRead(ENC_PAN_A);
  bool LSB = digitalRead(ENC_PAN_B);

  uint8_t encoded = (MSB << 1) | LSB;
  uint8_t sum = (lastEncodedPan << 2) | encoded;

  if (sum == 0b0001 || sum == 0b0111 || sum == 0b1110 || sum == 0b1000) encoderCountPan++;
  if (sum == 0b0010 || sum == 0b0100 || sum == 0b1101 || sum == 0b1011) encoderCountPan--;

  lastEncodedPan = encoded;
}

void IRAM_ATTR updateEncoderTilt() {
  bool MSB = digitalRead(ENC_TILT_A);
  bool LSB = digitalRead(ENC_TILT_B);

  uint8_t encoded = (MSB << 1) | LSB;
  uint8_t sum = (lastEncodedTilt << 2) | encoded;

  if (sum == 0b0001 || sum == 0b0111 || sum == 0b1110 || sum == 0b1000) encoderCountTilt++;
  if (sum == 0b0010 || sum == 0b0100 || sum == 0b1101 || sum == 0b1011) encoderCountTilt--;

  lastEncodedTilt = encoded;
}

// -------------------- PID-klasse --------------------
class PIDController {
  public:
    PIDController(float Kp, float Ki, float Kd) {
      this->Kp = Kp;
      this->Ki = Ki;
      this->Kd = Kd;
      this->previous_error = 0;
      this->integral = 0;
    }

    float compute(float process_variable, float setpoint, float dt) {
      float error = setpoint - process_variable;

      float P_out = Kp * error;
      float D_out = Kd * (error - previous_error) / dt;

      float output = P_out + Ki * integral + D_out;

      // Metning av ∆x for maksimal hastighet
      float maxDelta = maxSpeed * dt;
      output = constrain(output, -maxDelta, maxDelta);

      // Clamping anti-windup: Bare integrer hvis ikke mettet
      float unclamped_output = P_out + Ki * integral + D_out;
      if (output == unclamped_output) {
        integral += error * dt;
        integral = constrain(integral, -maxIntegral, maxIntegral);
      }

      previous_error = error;

      return output;
    }

    void setMaxIntegral(float maxInt) {
      maxIntegral = maxInt;
    }

  private:
    float Kp, Ki, Kd;
    float previous_error;
    float integral;
    float maxIntegral = 100.0;
};

PIDController pid_pan(1.0, 0.1, 0.0);
PIDController pid_tilt(1.0, 0.1, 0.0);

// -------------------- Setup --------------------
void setup() {
  Serial.begin(115200);

  servoPan.attach(servoPanPin);
  servoPan.write(servoPanPos);

  servoTilt.attach(servoTiltPin);
  servoTilt.write(servoTiltPos);

  pinMode(ENC_PAN_A, INPUT);
  pinMode(ENC_PAN_B, INPUT);

  bool A_pan = digitalRead(ENC_PAN_A);
  bool B_pan = digitalRead(ENC_PAN_B);
  lastEncodedPan = (A_pan << 1) | B_pan;

  attachInterrupt(digitalPinToInterrupt(ENC_PAN_A), updateEncoderPan, CHANGE);
  attachInterrupt(digitalPinToInterrupt(ENC_PAN_B), updateEncoderPan, CHANGE);

  pinMode(ENC_TILT_A, INPUT);
  pinMode(ENC_TILT_B, INPUT);

  bool A_tilt = digitalRead(ENC_TILT_A);
  bool B_tilt = digitalRead(ENC_TILT_B);
  lastEncodedTilt = (A_tilt << 1) | B_tilt;

  attachInterrupt(digitalPinToInterrupt(ENC_TILT_A), updateEncoderTilt, CHANGE);
  attachInterrupt(digitalPinToInterrupt(ENC_TILT_B), updateEncoderTilt, CHANGE);
}

// -------------------- Loop --------------------
void loop() {
  unsigned long now = millis();
  float dt = (now - lastTime) / 1000.0;
  lastTime = now;

  long count_pan, count_tilt;
  noInterrupts();
  count_pan = encoderCountPan;
  count_tilt = encoderCountTilt;
  interrupts();

  float PV_pan = -(360.0 * count_pan) / PPR;
  float PV_tilt = -(360.0 * count_tilt) / PPR;

  if (Serial.available()) {
    String input = Serial.readStringUntil('\n');
    input.trim();

    int commaIndex = input.indexOf(',');
    if (commaIndex > 0) {
      String pan = input.substring(0, commaIndex);
      String tilt = input.substring(commaIndex + 1);

      int panDeltaPixel = pan.toInt();
      int tiltDeltaPixel = tilt.toInt();

      float panDeltaAngle = ((float)(panDeltaPixel + 640) / 1280.0) * 110.0 - 55.0;
      float tiltDeltaAngle = ((float)(tiltDeltaPixel + 360) / 720.0) * 95.0 - 47.5;

      SP_pan = PV_pan + panDeltaAngle;
      SP_tilt = PV_tilt + tiltDeltaAngle;

      SP_pan = constrain(SP_pan, -80, 80);
      SP_tilt = constrain(SP_tilt, -80, 80);
      SP_tilt = map(SP_tilt, -80, 80, 80, -80);
    }
  }

  float delta_pan = pid_pan.compute(PV_pan, SP_pan, dt);
  float delta_tilt = pid_tilt.compute(PV_tilt, SP_tilt, dt);

  servoPanPos += delta_pan;
  servoPanPos = constrain(servoPanPos, 5.0, 165.0);

  servoTiltPos += delta_tilt;
  servoTiltPos = constrain(servoTiltPos, 20.0, 73.0);

  servoPan.write(servoPanPos);
  servoTilt.write(servoTiltPos);

  delay(50); // 20 Hz kontrollfrekvens
}
