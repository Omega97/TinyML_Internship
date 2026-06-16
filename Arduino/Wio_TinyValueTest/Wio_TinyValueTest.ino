/*
 * TinyML Neural Network - Forward Pass
 * 
 * A lightweight 3-layer Multi-Layer Perceptron (MLP) designed to run on 
 * resource-constrained devices like the Seeed Wio Terminal.
 * 
 * Architecture:
 *   Input      → 768 neurons
 *   Hidden 1   → 16 neurons (ReLU)
 *   Hidden 2   → 8  neurons (ReLU)
 *   Output     → 1  neuron  (tanh)
 * 
 * The model parameters are stored in "wio_weights.h"
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

  float input[768] = {0};   // test input (all zeros for now)
  float val = forward(input);

  Serial.print("Inferred value (test input): ");
  Serial.println(val, 6);
}

void loop() {}
