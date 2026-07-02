// 5-finger controller — drive-to-end, with calibrated EXTEND time (circular-reel safe).
// Retract = overdrive to grasp stall (no cal needed). Extend = drive open for calibrated time.
// CAL: W = begin. Press the key that CLOSES the hand once (records close dir + retracts all).
//      For each finger, press its EXTEND key to drive open, press E at FULL EXTENSION to record time:
//        F1 z  F2 x  F3 c  F4 v  F5 n
//      Press K when all fingers are calibrated.
// USE: o = OPEN hand   p = CLOSE hand
//      (camera later: g + 5 chars, c=retract o=extend, e.g. gococo)
//      per-finger: f + digit(1=index..5=thumb) + o/c (extend/retract), e.g. f1o f5c
// PuTTY @115200.

const int STBY = A3;
const int FING[5][3] = {  // {IN1, IN2, PWM}  index,middle,ring,pinky,thumb
  {  8,  7,  9 },{  6,  5, 10 },{  4,  2,  3 },{ 12, 13, 11 },{ A1, A2, A0 }
};

const int SPEED = 200;
const unsigned long RETRACT_MS = 3500;   // overdrive time to reach + hold grasp stall (then cut)
const unsigned long SAFETY_MS  = 5000;

int  closeKey = 0;
bool haveCloseDir = false;

long extendMs[5] = {0,0,0,0,0};   // calibrated open time per finger
bool calibrated = false;

// calibration: extending one finger and timing it
int  calFinger = -1;
unsigned long calStart = 0;

// runtime per-finger timed drive
int  fTarget[5]  = {0,0,0,0,0};   // +1 retract, -1 extend, 0 idle
unsigned long fStart[5] = {0,0,0,0,0};
unsigned long fDur[5]   = {0,0,0,0,0};

void driveFinger(int i, int d, int spd) {
  int in1 = FING[i][0], in2 = FING[i][1], pwm = FING[i][2];
  if (d > 0)      { digitalWrite(in1, HIGH); digitalWrite(in2, LOW);  analogWrite(pwm, spd); }
  else if (d < 0) { digitalWrite(in1, LOW);  digitalWrite(in2, HIGH); analogWrite(pwm, spd); }
  else            { digitalWrite(in1, LOW);  digitalWrite(in2, LOW);  analogWrite(pwm, 0); }
}
void allStop(){ for(int i=0;i<5;i++) driveFinger(i,0,0); }

void retractAll() {                       // drive all closed to the grasp stall
  Serial.println("Retracting all to closed reference...");
  unsigned long t = millis();
  while (millis() - t < RETRACT_MS) for (int i=0;i<5;i++) driveFinger(i, closeKey, SPEED);
  allStop();
}

void setup() {
  pinMode(STBY, OUTPUT); digitalWrite(STBY, HIGH);
  for (int i=0;i<5;i++) for (int j=0;j<3;j++) pinMode(FING[i][j], OUTPUT);
  allStop();
  Serial.begin(115200); delay(500);
  Serial.println("Press W, then the key that CLOSES the hand (SPACE or BACKSPACE).");
}

void loop() {
  if (Serial.available()) {
    char c = Serial.read();

    if (c == 'w' || c == 'W') {
      haveCloseDir = false; calibrated = false; calFinger = -1; allStop();
      Serial.println("Press SPACE or BACKSPACE = key that CLOSES the hand.");
    }
    else if (!haveCloseDir && c == ' ')        { closeKey=+1; haveCloseDir=true; retractAll();
        Serial.println("Close set. Calibrate EXTEND per finger: F1 z F2 x F3 c F4 v F5 n. Press key to extend, E at full open, K when done."); }
    else if (!haveCloseDir && (c==8||c==127))  { closeKey=-1; haveCloseDir=true; retractAll();
        Serial.println("Close set. Calibrate EXTEND per finger: F1 z F2 x F3 c F4 v F5 n. Press key to extend, E at full open, K when done."); }

    // ---- calibration phase ----
    else if (haveCloseDir && !calibrated) {
      int f=-1;
      switch(c){ case 'z':case'Z':f=0;break; case 'x':case'X':f=1;break;
                 case 'c':case'C':f=2;break; case 'v':case'V':f=3;break;
                 case 'n':case'N':f=4;break; }
      if (f>=0) { calFinger=f; calStart=millis();
                  Serial.print("Extending F"); Serial.print(f+1); Serial.println(" — press E at full extension."); }
      else if (c=='e'||c=='E') {
        if (calFinger>=0) {
          extendMs[calFinger] = millis()-calStart;
          driveFinger(calFinger,0,0);
          Serial.print("F"); Serial.print(calFinger+1); Serial.print(" extendMs="); Serial.println(extendMs[calFinger]);
          calFinger=-1;
        }
      }
      else if (c=='k'||c=='K') {
        calibrated=true; allStop();
        Serial.println("Calibration done. o = OPEN hand, p = CLOSE hand.");
      }
    }

    // ---- runtime: simple whole-hand keys ----
    else if (calibrated && (c=='o'||c=='O')) {        // OPEN whole hand (extend all)
      for (int i=0;i<5;i++){ fTarget[i]=-1; fStart[i]=millis(); fDur[i]=extendMs[i]; }
      Serial.println("OPENING hand...");
    }
    else if (calibrated && (c=='p'||c=='P')) {        // CLOSE whole hand (retract all)
      for (int i=0;i<5;i++){ fTarget[i]=+1; fStart[i]=millis(); fDur[i]=RETRACT_MS; }
      Serial.println("CLOSING hand...");
    }

    // ---- runtime gesture (for camera later) ----
    else if (calibrated && (c=='g'||c=='G')) {
      unsigned long t0=millis(); while(Serial.available()<5 && millis()-t0<300){}
      if (Serial.available()>=5) {
        for (int i=0;i<5;i++){
          char ch=Serial.read();
          if      (ch=='c'||ch=='C'){ fTarget[i]=+1; fStart[i]=millis(); fDur[i]=RETRACT_MS; }
          else if (ch=='o'||ch=='O'){ fTarget[i]=-1; fStart[i]=millis(); fDur[i]=extendMs[i]; }
        }
        Serial.println("Gesture applied.");
      }
    }

    // ---- runtime: individual finger control (camera bridge) ----
    else if (calibrated && (c=='f'||c=='F')) {
      unsigned long t0=millis(); while(Serial.available()<2 && millis()-t0<300){}
      if (Serial.available()>=2) {
        char fch=Serial.read();      // '1'..'5' : F1 index .. F5 thumb
        char dch=Serial.read();      // 'o' extend, 'c' retract
        int i = fch - '1';
        if (i>=0 && i<5) {
          if      (dch=='c'||dch=='C'){ fTarget[i]=+1; fStart[i]=millis(); fDur[i]=RETRACT_MS; }
          else if (dch=='o'||dch=='O'){ fTarget[i]=-1; fStart[i]=millis(); fDur[i]=extendMs[i]; }
          Serial.print("F"); Serial.print(i+1);
          Serial.println((dch=='c'||dch=='C') ? " retract" : " extend");
        }
      }
    }
  }

  // ---- calibration: drive the finger being timed ----
  if (haveCloseDir && !calibrated && calFinger>=0) {
    driveFinger(calFinger, -closeKey, SPEED);    // extend direction
    if (millis()-calStart > SAFETY_MS) driveFinger(calFinger,0,0);  // safety cap
  }

  // ---- runtime: per-finger timed drive ----
  if (calibrated) {
    for (int i=0;i<5;i++){
      if (fTarget[i]!=0) {
        if (millis()-fStart[i] < fDur[i]) {
          int phys = (fTarget[i]>0) ? closeKey : -closeKey;
          driveFinger(i, phys, SPEED);
        } else { driveFinger(i,0,0); fTarget[i]=0; }
      }
    }
  }
}