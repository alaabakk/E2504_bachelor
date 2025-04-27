#include <ESP32Servo.h>

#define ENC_ROT_A 32
#define ENC_ROT_B 33
#define ENC_TILT_A 34
#define ENC_TILT_B 35
#define PPR 1024

#define SERVO1_PIN 18
#define SERVO2_PIN 19
#define SERVO_PWM_FREQ 50
#define PWM_RESOLUTION 16

Servo myservo1;
Servo myservo2;

volatile long encoderCountRot = 0;
volatile long encoderCountTilt = 0;

void IRAM_ATTR encoderFuncRot() {
  if (digitalRead(ENC_ROT_B) == LOW) {
    if (encoderCountRot < 512) {
      encoderCountRot++;
    }
  } else {
    if (encoderCountRot > 0) { 
      encoderCountRot--;
    }
  }
}

void IRAM_ATTR encoderFuncTilt() {
  if (digitalRead(ENC_TILT_B) == LOW) {
    if (encoderCountTilt > 0) {
    encoderCountTilt--;
    }
  } else {
    if (encoderCountTilt < 512) {
    encoderCountTilt++;
    }
  }
}

void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600);

  pinMode(ENC_ROT_A, INPUT);
  pinMode(ENC_ROT_B, INPUT);
  attachInterrupt(digitalPinToInterrupt(ENC_ROT_A), encoderFuncRot, RISING);

  pinMode(ENC_TILT_A, INPUT);
  pinMode(ENC_TILT_B, INPUT);
  attachInterrupt(digitalPinToInterrupt(ENC_TILT_A), encoderFuncTilt, RISING);

  myservo1.attach(SERVO1_PIN);
  myservo2.attach(SERVO2_PIN);
  //myservo1.attach(SERVO1_PIN, 500, 2400);
  //myservo2.attach(SERVO2_PIN, 500, 2400);
}

void loop() {
  // put your main code here, to run repeatedly:
static unsigned long lastPrint = 0;

  if (millis() - lastPrint >= 10) {
    lastPrint = millis();

    // Read counts safely
    long count1, count2;
    noInterrupts();
    count1 = encoderCountRot;
    count2 = encoderCountTilt;
    interrupts();

    // Convert encoder counts to angle (clamp between 0–180)
    float angleRot = (360.0 * count1) / PPR;
    float angleTilt = (360.0 * count2) / PPR;

    myservo1.write(angleRot);
    myservo2.write(angleTilt);

    // Print both angles
    Serial.print("Encoder Rot: ");
    Serial.print(count1);
    Serial.print(" | ");
    Serial.print(angleRot, 2);
    Serial.print("°   ||   ");

    Serial.print("Encoder Tilt: ");
    Serial.print(count2);
    Serial.print(" | ");
    Serial.print(angleTilt, 2);
    Serial.println("°");
  }
}