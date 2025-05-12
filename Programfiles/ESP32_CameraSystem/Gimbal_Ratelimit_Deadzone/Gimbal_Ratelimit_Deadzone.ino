#include <Arduino.h>
#include <ESP32Servo.h>

// -------------------- Pins --------------------
static const int servoPanPin = 18;
static const int servoTiltPin = 19;

// -------------------- Servo-innstillinger --------------------
Servo servoPan;
Servo servoTilt;

int servoPanMiddle = 90;
float servoPanPos = servoPanMiddle;
int servoPanMin = 5;
int servoPanMax = 165;

int servoTiltMiddle = 56;
float servoTiltPos = servoTiltMiddle;
int servoTiltMin = 20;
int servoTiltMax = 73;

// -------------------- Kamera og kontrollparametere --------------------
float camera_width = 1280.0;
float camera_height = 720.0;
float camera_H_fov = 110.0; // horisontal FOV i grader
float camera_V_fov = 95.0;  // vertikal FOV i grader

float maxSpeed = 50.0; // grader per sekund
float deadzone = 3.0;  // grader (ignorér små vinkelavvik)

// -------------------- Tidshåndtering --------------------
unsigned long lastTime = 0;

void setup() {
  Serial.begin(115200);

  servoPan.attach(servoPanPin);
  servoTilt.attach(servoTiltPin);

  // Sett servoene til midtstilling
  servoPan.write(servoPanPos);
  servoTilt.write(servoTiltPos);
  delay(3000);
}

void loop() {
  unsigned long now = millis();
  float dt = (now - lastTime) / 1000.0;
  lastTime = now;

  if (Serial.available()) {
    String input = Serial.readStringUntil('\n');
    input.trim();

    if (input == "q") {
      // Tilbakestill til midtposisjon
      servoPanPos = servoPanMiddle;
      servoTiltPos = servoTiltMiddle;
    } else {
      int commaIndex = input.indexOf(',');
      if (commaIndex > 0) {
        String pan = input.substring(0, commaIndex);
        String tilt = input.substring(commaIndex + 1);

        int panDeltaPixel = pan.toInt();
        int tiltDeltaPixel = tilt.toInt();

        // Konverter pikselavvik til vinkel
        float panDeltaAngle = map(panDeltaPixel, 0, camera_width, camera_H_fov / 2, -camera_H_fov / 2);
        float tiltDeltaAngle = map(tiltDeltaPixel, 0, camera_height, -camera_V_fov / 2, camera_V_fov / 2);

        float maxDelta = maxSpeed * dt;

        // PAN-kontroll med deadzone og ratebegrensning
        if (abs(panDeltaAngle) > deadzone) {
          float targetPan = servoPanPos + panDeltaAngle;
          float panStep = constrain(targetPan - servoPanPos, -maxDelta, maxDelta);
          servoPanPos += panStep;
        }

        // TILT-kontroll med deadzone og ratebegrensning
        if (abs(tiltDeltaAngle) > deadzone) {
          float targetTilt = servoTiltPos + tiltDeltaAngle;
          float tiltStep = constrain(targetTilt - servoTiltPos, -maxDelta, maxDelta);
          servoTiltPos += tiltStep;
        }
      }
    }
  }

  // Begrens posisjonene til servoens fysiske grenser
  servoPanPos = constrain(servoPanPos, servoPanMin, servoPanMax);
  servoTiltPos = constrain(servoTiltPos, servoTiltMin, servoTiltMax);

  // Skriv til servoene
  servoPan.write(servoPanPos);
  servoTilt.write(servoTiltPos);

  delay(50); // 20 Hz kontrollfrekvens
}
