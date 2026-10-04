/*
 * HYDRA data logger - Arduino (Uno/Nano/Mega) or ESP32.  READ-ONLY: this firmware has NO actuator outputs.
 *
 * Channels (1 Hz CSV over serial, 115200 baud):
 *   t_s, T1_C (liquid, DS18B20), T2_C (wall, DS18B20), P_bar_g (analog transducer), flow_L_min (pulse flow meter),
 *   I_A (INA219 shunt), V_V (INA219 bus voltage)
 *
 * Libraries: OneWire, DallasTemperature, Adafruit INA219 (optional: define USE_INA219).
 * Wiring: see docs/HARDWARE.md.  A physical pressure-relief device and emergency stop MUST exist independently of
 * this logger and of any software.
 */
#include <OneWire.h>
#include <DallasTemperature.h>
#ifdef USE_INA219
#include <Adafruit_INA219.h>
Adafruit_INA219 ina;
#endif

// ---- pins (Uno/Nano defaults; for ESP32 use e.g. ONE_WIRE=4, PRESS=34, FLOW=27) -------------------------------
#define ONE_WIRE_PIN 2
#define PRESSURE_PIN A0
#define FLOW_PIN 3            // interrupt-capable pin on the Uno/Nano

// ---- calibration (EDIT after calibrating against a reference gauge / measuring cylinder) ----------------------
const float P_V_MIN = 0.5f;   // transducer output at 0 bar(g) [V]
const float P_V_MAX = 4.5f;   // transducer output at full scale [V]
const float P_FS_BAR = 5.0f;  // full scale [bar(g)]
const float ADC_VREF = 5.0f;  // 3.3 for ESP32 (ESP32 ADC is non-linear: add a lookup table or use an ADS1115)
const float ADC_MAX = 1023.0f;  // 4095 for ESP32
const float PULSES_PER_LITRE = 450.0f;  // YF-S201 ~450 pulses/L; CALIBRATE with a gas syringe
const float SENSOR_HZ = 1.0f;

OneWire oneWire(ONE_WIRE_PIN);
DallasTemperature ds(&oneWire);
volatile unsigned long pulses = 0;
unsigned long lastMs = 0, lastPulses = 0;

void onPulse() { pulses++; }

void setup() {
  Serial.begin(115200);
  ds.begin();
  ds.setResolution(11);       // 0.125 C steps, ~375 ms conversion
  ds.setWaitForConversion(false);
  pinMode(FLOW_PIN, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(FLOW_PIN), onPulse, FALLING);
#ifdef USE_INA219
  ina.begin();
#endif
  Serial.println("t_s,T1_C,T2_C,P_bar_g,flow_L_min,I_A,V_V");
  ds.requestTemperatures();
  lastMs = millis();
}

float readPressureBar() {
  float acc = 0;
  for (int i = 0; i < 16; i++) acc += analogRead(PRESSURE_PIN);   // average to reduce noise
  float v = (acc / 16.0f) / ADC_MAX * ADC_VREF;
  float p = (v - P_V_MIN) / (P_V_MAX - P_V_MIN) * P_FS_BAR;
  return p;                                                          // gauge pressure, can be slightly negative
}

void loop() {
  unsigned long now = millis();
  if (now - lastMs >= (unsigned long)(1000.0f / SENSOR_HZ)) {
    float dt = (now - lastMs) / 1000.0f;
    lastMs = now;
    float t1 = ds.getTempCByIndex(0);   // liquid
    float t2 = ds.getTempCByIndex(1);   // wall
    ds.requestTemperatures();           // non-blocking: result used in the next cycle (1 s lag, documented)
    noInterrupts(); unsigned long pc = pulses; interrupts();
    float flow = ((pc - lastPulses) / PULSES_PER_LITRE) / (dt / 60.0f);   // L/min at meter conditions
    lastPulses = pc;
    float cur = NAN, volt = NAN;
#ifdef USE_INA219
    cur = ina.getCurrent_mA() / 1000.0f;
    volt = ina.getBusVoltage_V();
#endif
    Serial.print(now / 1000.0f, 3); Serial.print(',');
    Serial.print(t1 == DEVICE_DISCONNECTED_C ? NAN : t1, 2); Serial.print(',');
    Serial.print(t2 == DEVICE_DISCONNECTED_C ? NAN : t2, 2); Serial.print(',');
    Serial.print(readPressureBar(), 3); Serial.print(',');
    Serial.print(flow, 3); Serial.print(',');
    Serial.print(cur, 3); Serial.print(',');
    Serial.println(volt, 3);
  }
}
