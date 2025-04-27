#include <Servo.h>

Servo myservo1;
Servo myservo2;

const int analogInPin0 = A0;  // Analog input pin that the potentiometer is attached to
const int analogInPin1 = A1;


int Servo1Pos = 0;        // value output to the PWM (analog out)
int Servo2Pos = 0; 


void setup() {
  myservo1.attach(9);
  myservo2.attach(10);

  Serial.begin(9600);
}

void loop() {
  if (Serial.available()) {
    String input = Serial.readStringUntil('\n'); // Read until newline
    
    int commaIndex = input.indexOf(',');
    if (commaIndex > 0) {
      String s1 = input.substring(0, commaIndex);
      String s2 = input.substring(commaIndex + 1);

      Servo1Pos = map(s1.toInt(), 0, 1280, 145, 35);
      Servo2Pos = map(s2.toInt(), 0, 1280, 145, 35);

      myservo1.write(Servo1Pos);
      myservo2.write(Servo2Pos);
    }
  }
}