/*
 * TinyML Neural Network - Forward Pass
 * Lightweight 3-layer MLP for Seeed Wio Terminal.
 * Architecture (Dynamic): Configured via the active weights header file.
 * The model parameters are stored in the file named by WEIGHTS_FILE below.
 */

// Define the weights filename here so it can be reused in print statements.
// Swap the active line (uncomment one, comment the other) to switch models easily.
// Both headers must be present in this folder and contain their respective layer macros.
// #define WEIGHTS_FILE "wio_int8_weights_nano.h"     // nano model (~12.5k params)
#define WEIGHTS_FILE "wio_int8_weights_tiny.h"   // tiny model (~25k params)

#include WEIGHTS_FILE
#include "fen_input.h"
#include <math.h>
#include <avr/pgmspace.h>

#include <TFT_eSPI.h>
#include <SPI.h>

TFT_eSPI tft = TFT_eSPI();

// Int8 quantized forward pass (weights from the file named by WEIGHTS_FILE, scales applied on-the-fly)
// Input x is float 0/1 (from fen_input.h). We treat input_scale = 1.0
// This approximates the original float32 model while using 4x less weight memory.
float forward(const float* x) {
  
  // --- LAYER 1 (Input -> Hidden 1) ---
  float h1[FC1_OUT_DIM];
  for (int i = 0; i < FC1_OUT_DIM; i++) {
    int32_t acc = 0;
    for (int j = 0; j < FC1_IN_DIM; j++) {
      int8_t wq = pgm_read_byte(&fc1_w[i * FC1_IN_DIM + j]);
      int8_t xq = (x[j] > 0.5f ? 1 : 0);
      acc += (int32_t)wq * xq;
    }
    float sum = (float)acc * fc1_w_scale * input_scale;
    int8_t bq = pgm_read_byte(&fc1_b[i]);
    sum += (float)bq * fc1_b_scale;
    h1[i] = (sum > 0.0f) ? sum : 0.0f; // ReLU
  }

  // --- LAYER 2 (Hidden 1 -> Hidden 2) ---
  float h2[FC2_OUT_DIM];
  for (int i = 0; i < FC2_OUT_DIM; i++) {
    int32_t acc = 0;
    for (int j = 0; j < FC2_IN_DIM; j++) {
      int8_t wq = pgm_read_byte(&fc2_w[i * FC2_IN_DIM + j]);
      int8_t xq = (h1[j] > 0.5f ? 1 : 0); // requantize activation roughly
      acc += (int32_t)wq * xq;
    }
    float sum = (float)acc * fc2_w_scale; // input_scale for this layer approximated
    int8_t bq = pgm_read_byte(&fc2_b[i]);
    sum += (float)bq * fc2_b_scale;
    h2[i] = (sum > 0.0f) ? sum : 0.0f; // ReLU
  }

  // --- LAYER 3 (Hidden 2 -> Output) ---
  int32_t acc = 0;
  for (int j = 0; j < FC3_IN_DIM; j++) {
    int8_t wq = pgm_read_byte(&fc3_w[j]);
    int8_t xq = (h2[j] > 0.5f ? 1 : 0);
    acc += (int32_t)wq * xq;
  }
  float out = (float)acc * fc3_w_scale;
  int8_t bq = pgm_read_byte(&fc3_b[0]);
  out += (float)bq * fc3_b_scale;
  return tanhf(out);
}

void setup() {
  Serial.begin(115200);
  while (!Serial);

  // Print filename to Serial Monitor
  Serial.print("Loaded Weights File: ");
  Serial.println(WEIGHTS_FILE);

  // Also print first 3 weights to Serial (to cross-check with LCD)
  // Cast via int8_t first so negative values print correctly (pgm_read_byte returns uint8_t)
  Serial.print("First 3 weights from fc1_w: ");
  Serial.print((int)(int8_t)pgm_read_byte(&fc1_w[0]));
  Serial.print(",");
  Serial.print((int)(int8_t)pgm_read_byte(&fc1_w[1]));
  Serial.print(",");
  Serial.println((int)(int8_t)pgm_read_byte(&fc1_w[2]));

  tft.init();
  tft.setRotation(3);
  tft.fillScreen(TFT_BLACK);
  tft.setTextColor(TFT_WHITE);

  // Print filename to TFT display at the bottom (y=70)
  tft.setTextSize(1); // Smaller text size for long filenames
  tft.setCursor(10, 70);
  tft.print("Weights: ");
  tft.print(WEIGHTS_FILE);

  // Print first 3 weights (from fc1_w) in small text to verify we are actually loading/using the int8 data
  tft.setCursor(10, 85);
  tft.print("W[0:2]:");
  tft.print((int)(int8_t)pgm_read_byte(&fc1_w[0]));
  tft.print(",");
  tft.print((int)(int8_t)pgm_read_byte(&fc1_w[1]));
  tft.print(",");
  tft.print((int)(int8_t)pgm_read_byte(&fc1_w[2]));

  // Label once (static)
  tft.setTextSize(2); // Reset to original size for performance metric
  tft.setCursor(10, 10);
  tft.print("Inferred value:");
}

void loop() {
  // Stress test: forward as fast as possible, report evals/sec every 1s
  float val = forward(input);
  static long evals = 0;
  static unsigned long lastReport = 0;
  evals++;
  unsigned long now = millis();

  if (now - lastReport >= 1000) {
    float million_evals = evals / 1000000.0f;

    // Serial output
    Serial.print("Evals/s: ");
    Serial.print(million_evals, 2);
    Serial.println("M");

    // TFT display update
    tft.fillRect(10, 40, 220, 16, TFT_BLACK);
    tft.setCursor(10, 40);
    tft.print("Evals/s: ");
    tft.print(million_evals, 4);
    tft.println("M");

    evals = 0;
    lastReport = now;
  }
}
