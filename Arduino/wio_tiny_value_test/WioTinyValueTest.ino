/*
  WioTinyValueTest.ino
  Minimal sketch to run the UltraTinyValueMLP on Wio Terminal using extracted weights.
  
  1. Copy wio_weights.h into this sketch folder (next to this .ino).
  2. Upload to Wio Terminal (board: Seeed Wio Terminal).
  3. Open Serial Monitor at 115200 baud (make sure no external battery is plugged for USB power).
  
  This computes the value for a test input (all zeros) and prints it.
  Later we will use a real board state (FEN -> 768 features).
*/

#include "wio_weights.h"
#include <math.h>

float forward(const float* x) {
  float h1[16];
  for (int i = 0; i < 16; i++) {
    float sum = fc1_b[i];
    for (int j = 0; j < 768; j++) {
      sum += x[j] * fc1_w[i * 768 + j];
    }
    h1[i] = (sum > 0.0f) ? sum : 0.0f;   // ReLU
  }

  float h2[8];
  for (int i = 0; i < 8; i++) {
    float sum = fc2_b[i];
    for (int j = 0; j < 16; j++) {
      sum += h1[j] * fc2_w[i * 16 + j];
    }
    h2[i] = (sum > 0.0f) ? sum : 0.0f;   // ReLU
  }

  float out = fc3_b[0];
  for (int j = 0; j < 8; j++) {
    out += h2[j] * fc3_w[j];
  }
  return tanhf(out);
}

void setup() {
  Serial.begin(115200);
  while (!Serial);

  float input[768] = {0};   // test input (all zeros)
  float val = forward(input);

  Serial.print("Inferred value (test input): ");
  Serial.println(val, 6);
}

void loop() {
  // nothing
}
