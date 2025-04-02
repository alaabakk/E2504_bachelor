#define ENC_ROT_A 32
#define ENC_ROT_B 33

#define ENC_TILT_A 34
#define ENC_TILT_B 35

#define PPR 1024

volatile long encoderCountRot = 0;
volatile long encoderCountTilt = 0;

void IRAM_ATTR encoderFuncRot() {
  if (digitalRead(ENC_ROT_B) == LOW) {
    encoderCountRot++;
  } else {
    encoderCountRot--;
  }
}

void IRAM_ATTR encoderFuncTilt() {
  if (digitalRead(ENC_TILT_B) == LOW) {
    encoderCountTilt--;
  } else {
    encoderCountTilt++;
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

    float angleRot = (360.0 * count1) / PPR;
    float angleTilt = (360.0 * count2) / PPR;

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