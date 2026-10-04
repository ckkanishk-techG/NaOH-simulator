# HYDRA hardware logging guide (read-only)

**Safety first.** A mechanical pressure-relief device and a physical emergency stop must exist and work
independently of any software. The supplied firmware and the live twin are *read-only*: there is no actuator code.
Aluminium/NaOH produces heat, caustic mist and flammable hydrogen: supervise, ventilate, wear PPE.

## Parts
| Quantity | Sensor | Interface |
|---|---|---|
| Liquid T | DS18B20 (stainless probe) in a PTFE/PE well | 1-Wire (4.7 kΩ pull-up) |
| Wall T | DS18B20 taped to the HDPE wall | same 1-Wire bus |
| (alt. T > 100 °C) | K-type thermocouple + MAX31855 | SPI |
| Gauge pressure | 0–5 bar(g) 0.5–4.5 V transducer (chemically compatible, e.g. PTFE/316 wetted) | analog A0 (use ADS1115 on ESP32) |
| H2 flow | pulse gas flow meter (e.g. YF-S201 is for liquids: use a gas-rated meter or a water-displacement counter) | interrupt pin 3 |
| Stack I, V | INA219 on the stack output | I²C |

## Wiring (Arduino Uno/Nano)
```
DS18B20 data -> D2  (+ 4.7k to 5V)        transducer signal -> A0 (supply 5V/GND)
flow pulse   -> D3  (INPUT_PULLUP)        INA219 SDA/SCL -> A4/A5 (shunt in series with the stack output)
ESP32: ONE_WIRE=GPIO4, PRESSURE=GPIO34 (ADC1, 3.3 V divider, 12-bit, non-linear -> ADS1115 recommended), FLOW=GPIO27
```
Isolate the transducer wiring from the caustic splash zone; keep electronics out of the hydrogen-rich volume.

## Calibration (do this before trusting the twin)
* Pressure: compare with a reference gauge at 3 points, edit `P_V_MIN`, `P_V_MAX`, `P_FS_BAR`.
* Flow: collect a known volume with a gas syringe / measuring cylinder, edit `PULSES_PER_LITRE`.
* Temperature: ice-point and warm-bath check; DS18B20 accuracy ~0.5 °C (the twin uses `sd_T`).
Use `hydra.inference.corrections` for water-displacement vapour/hydrostatic/dissolution corrections.

## Data format
`t_s,T1_C,T2_C,P_bar_g,flow_L_min,I_A,V_V` at 1 Hz (CSV, `nan` for missing). The twin ingests it from serial,
a streaming CSV file or Modbus (`hydra.hardware.sources`).
