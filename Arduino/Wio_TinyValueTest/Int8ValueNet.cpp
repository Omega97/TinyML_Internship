#include "Int8ValueNet.h"

#include "config.h"
#include WEIGHTS_FILE

#include <avr/pgmspace.h>
#include <math.h>

float Int8ValueNet::evaluate(const float* x) const {
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
    h1[i] = (sum > 0.0f) ? sum : 0.0f;
  }

  float h2[FC2_OUT_DIM];
  for (int i = 0; i < FC2_OUT_DIM; i++) {
    int32_t acc = 0;
    for (int j = 0; j < FC2_IN_DIM; j++) {
      int8_t wq = pgm_read_byte(&fc2_w[i * FC2_IN_DIM + j]);
      int8_t xq = (h1[j] > 0.5f ? 1 : 0);
      acc += (int32_t)wq * xq;
    }
    float sum = (float)acc * fc2_w_scale;
    int8_t bq = pgm_read_byte(&fc2_b[i]);
    sum += (float)bq * fc2_b_scale;
    h2[i] = (sum > 0.0f) ? sum : 0.0f;
  }

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

int8_t Int8ValueNet::weightAt(int idx) {
  return (int8_t)pgm_read_byte(&fc1_w[idx]);
}

void Int8ValueNet::printArchitecture() {
  Serial.print(FC1_IN_DIM); Serial.print("→"); Serial.print(FC1_OUT_DIM);
  Serial.print("→"); Serial.print(FC2_OUT_DIM);
  Serial.print("→"); Serial.println(FC3_OUT_DIM);
}