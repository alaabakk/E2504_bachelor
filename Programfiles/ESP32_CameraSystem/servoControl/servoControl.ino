#include <Arduino.h>
#include <ESP32Servo.h>
 
// -------------------- Pins --------------------
static const int servoPanPin = 18;
static const int servoTiltPin = 19;
 
 
// -------------------- Globals --------------------
Servo servoPan;
Servo servoTilt;
 
volatile long encoderCountPan = 0;
volatile long encoderCountTilt = 0;
 
volatile uint8_t lastEncodedPan = 0;
volatile uint8_t lastEncodedTilt = 0;
 
float maxSpeed = 50.0; // grader per sekund ved 100% pådrag
 
int servoPanMiddle = 90;
float servoPanPos = servoPanMiddle;
int servoPanMin = 5;
int servoPanMax = 165;
 
int servoTiltMiddle = 56;
float servoTiltPos = servoTiltMiddle;
int servoTiltMin = 20;
int servoTiltMax = 73;
 
 
float camera_width = 1280.0; // Kamerabredde i pixel
float camera_height = 720.0; // Kamerahøyde i pixel
float camera_H_fov = 110.0; // Kamerabredde i grader (horisontal FOV)
float camera_V_fov = 95.0; // Kamerahøyde i grader (vertical FOV)
 
float SP_pan;
float SP_tilt;
float e_pan = 0;
float e_tilt = 0;
 
unsigned long lastTime = 0;
 

// -------------------- PID-klasse --------------------
class PIController {
    public:
      PIController(float Kp, float Ki) {
        this->Kp = Kp;
        this->Ki = Ki;
        this->integral = 0;
      }
 
      float compute(float error, float dt) {
        if (abs(error) < 3) {
          return 0;
        }
        float P_out = Kp * error;
        float I_out = Ki * integral;
 
        float output = P_out + I_out;
 
        // Metning av ∆x for maksimal hastighet
        float maxDelta = maxSpeed * dt;
        output = constrain(output, -maxDelta, maxDelta);
 
        // Clamping anti-windup: Bare integrer hvis ikke mettet
        float unclamped_output = P_out + I_out;
        if (output == unclamped_output) {
          integral += error * dt;
          integral = constrain(integral, -maxIntegral, maxIntegral);
        }
 
        return P_out;
      }
 
      void setMaxIntegral(float maxInt) {
        maxIntegral = maxInt;
      }
 
    private:
      float Kp, Ki;
      float integral;
      float maxIntegral = 100.0;
  };
 
PIController pid_pan(0.1, 0.1);
PIController pid_tilt(0.1, 0.1);
 
// -------------------- Setup --------------------
void setup() {
  Serial.begin(115200);
 
  servoPan.attach(servoPanPin);
  servoTilt.attach(servoTiltPin);
 
  // Setter servoene til midtsilling
  servoPan.write(servoPanPos);
  servoTilt.write(servoTiltPos);
  delay(3000);
 
}
 
// -------------------- Loop --------------------
void loop() {
  unsigned long now = millis();
  float dt = (now - lastTime) / 1000.0;
  lastTime = now;
 
  // Henter innkommende data fra Serial Monitor
  if (Serial.available()) {
    String input = Serial.readStringUntil('\n');
    input.trim();
 
    if (input == "q") {
      servoPanPos = servoPanMiddle;
      servoTiltPos = servoTiltMiddle;
      e_pan = 0;
      e_tilt = 0;


    } else {
      int commaIndex = input.indexOf(',');
      if (commaIndex > 0) {
        String pan = input.substring(0, commaIndex); // Henter ut pan-verdien
        String tilt = input.substring(commaIndex + 1); // Henter ut tilt-verdien
  
        int panDeltaPixel = pan.toInt();
        int tiltDeltaPixel = tilt.toInt();
  
        float panDeltaAngle = map(panDeltaPixel, 0, camera_width, camera_H_fov / 2, -camera_H_fov / 2);
        float tiltDeltaAngle = map(tiltDeltaPixel, 0, camera_height, -camera_V_fov / 2, camera_V_fov / 2);
  
        e_pan = panDeltaAngle;
        e_tilt = tiltDeltaAngle;
      
      }
    }
  }
 
  float delta_pan = pid_pan.compute(e_pan, dt);
  float delta_tilt = pid_tilt.compute(e_tilt, dt);
 
  servoPanPos += delta_pan;
  servoTiltPos += delta_tilt;
 
  servoPanPos = constrain(servoPanPos, servoPanMin, servoPanMax);
  servoTiltPos = constrain(servoTiltPos, servoTiltMin, servoTiltMax);
 
  servoPan.write(servoPanPos);
  servoTilt.write(servoTiltPos);
 
  delay(50); // 20 Hz kontrollfrekvens
}