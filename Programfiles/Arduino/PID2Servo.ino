#include <Arduino.h>
#include <ESP32Servo.h>

// -------------------- Pins --------------------
static const int servoPanPin = 18;
static const int servoTiltPin = 19;

#define ENC_PAN_A 32
#define ENC_PAN_B 33

#define ENC_TILT_A 35
#define ENC_TILT_B 34

#define PPR 4096  // 1024 PPR encoder, 4x kvadratur

// -------------------- Globals --------------------
Servo servoPan;
Servo servoTilt;

volatile long encoderCountPan = 0;
volatile long encoderCountTilt = 0;

volatile uint8_t lastEncodedPan = 0;
volatile uint8_t lastEncodedTilt = 0;

float maxSpeed = 90.0; // grader per sekund ved 100% pådrag
float servoPanPos = 90.0; // startposisjon i midten
float servoTiltPos = 90.0;
float SP_pan;
float SP_tilt;
float PV_pan;
float PV_tilt;
float U_pan;
float U_tilt;


unsigned long lastTime = 0;

// -------------------- Interrupt --------------------
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

// -------------------- Servo control --------------------
void controlPanServo(float U_pan, float dt ) {
  float deltaPos = (U_pan / 100.0) * maxSpeed * dt;
  servoPanPos += deltaPos;
  servoPanPos = constrain(servoPanPos, 10.0, 170.0);
}

void controlTiltServo(float U_tilt, float dt ) {
  float deltaPos = (U_tilt / 100.0) * maxSpeed * dt;
  servoTiltPos += deltaPos;
  servoTiltPos = constrain(servoTiltPos, 10.0, 170.0);
}



// -------------------- PID-klasse --------------------
class PIDController {
  public:
    PIDController(float Kp, float Ki, float Kd) {
      this->Kp = Kp;
      this->Ki = Ki;
      this->Kd = Kd;
      this->setpoint = 0;
      this->previous_error = 0;
      this->integral = 0;
    }

    void setSetpoint(float sp) {
      setpoint = sp;
    }

    float compute(float process_variable, float setpoint, float dt) {
      this->setpoint = setpoint;
      float error = setpoint - process_variable;

      float P_out = Kp * error;
      integral += error * dt;

      float max_integral = 1000.0;
      if (integral > max_integral) integral = max_integral;
      else if (integral < -max_integral) integral = -max_integral;

      float I_out = Ki * integral;

      float output;
      if (abs(error) < 1.0) {
        output = 0;
      } else {
        output = P_out; // + I_out;
        output = constrain(output, -100, 100);
      }

      return output;
    }

  private:
    float Kp, Ki, Kd;
    float setpoint;
    float previous_error;
    float integral;
};

PIDController pid_pan(1.0, 0.1, 0.0); // Juster Kp, Ki, Kd ved behov
PIDController pid_tilt(1.0, 0.1, 0.0);


// -------------------- Setup --------------------
void setup() {
  Serial.begin(115200);

  servoPan.attach(servoPanPin);
  servoPan.write(servoPanPos); // Midtposisjon ved start

  servoTilt.attach(servoTiltPin);
  servoTilt.write(servoTiltPos);


// -------------------- Encoder 1 --------------------
  pinMode(ENC_PAN_A, INPUT);
  pinMode(ENC_PAN_B, INPUT);

  bool A_pan = digitalRead(ENC_PAN_A);
  bool B_pan = digitalRead(ENC_PAN_B);
  lastEncodedPan = (A_pan << 1) | B_pan;

  attachInterrupt(digitalPinToInterrupt(ENC_PAN_A), updateEncoderPan, CHANGE);
  attachInterrupt(digitalPinToInterrupt(ENC_PAN_B), updateEncoderPan, CHANGE);

// -------------------- Encoder 2 --------------------
  pinMode(ENC_TILT_A, INPUT);
  pinMode(ENC_TILT_B, INPUT);

  bool A_tilt = digitalRead(ENC_TILT_A);
  bool B_tilt = digitalRead(ENC_TILT_B);
  lastEncodedPan = (A_tilt << 1) | B_tilt;

  attachInterrupt(digitalPinToInterrupt(ENC_TILT_A), updateEncoderTilt, CHANGE);
  attachInterrupt(digitalPinToInterrupt(ENC_TILT_B), updateEncoderTilt, CHANGE);

}

// -------------------- Loop --------------------
void loop() {
  unsigned long now = millis();
  float dt = (now - lastTime) / 1000.0;  // i sekunder
  lastTime = now;

  // Les encoderposisjon
  long count;
  noInterrupts();
  count = encoderCountPan;
  interrupts();
  float PV_pan = -(360.0 * count) / PPR;

  // Les ønsket posisjon fra potmeter
  if (Serial.available()) {
    String input = Serial.readStringUntil('\n');  // Read until newline
    input.trim();  

    int commaIndex = input.indexOf(',');

    if (commaIndex > 0) {
      String pan = input.substring(0, commaIndex);
      String tilt = input.substring(commaIndex + 1);

      int panDeltaPixel = pan.toInt();
      int tiltDeltaPixel = tilt.toInt();

      float panDeltaAngle = map(panDeltaPixel, -640, 640, -55, 55);
      float tiltDeltaAngle = map(tiltDeltaPixel, -360, 360, -47.5, 47.5);

      SP_pan = PV_pan + panDeltaAngle;
      SP_tilt = PV_tilt + tiltDeltaAngle;

      SP_pan = constrain(SP_pan, -80, 80);
      SP_tilt = constrain(SP_tilt, -80, 80);    
    }

  }

  // PID
  U_pan = pid_pan.compute(PV_pan, SP_pan, dt); // -100 til +100 prosent
  U_tilt = pid_tilt.compute(PV_pan, SP_pan, dt); // -100 til +100 prosent
  // Simulert hastighetsstyring

  controlPanServo(U_pan, dt);
  controlTiltServo(U_tilt, dt);
  servoPan.write(servoPanPos);




  delay(50);  // gir 20Hz kontrollfrekvens (kan justeres)
}
