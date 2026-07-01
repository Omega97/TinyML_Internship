#include "Int8ValueNet.h"

#include "config.h"
#include WEIGHTS_FILE

#include <avr/pgmspace.h>
#include <math.h>
#include <stdint.h>

volatile uint32_t g_evalCalls = 0;

// Max hidden dims across nano→huge headers; static avoids stack overflow on huge.
static float s_h1[512];
static float s_h2[64];

// input[] lives in PROGMEM (fen_input.h). Must not use plain x[j] — and must
// not let -Os constant-fold the whole network from const input + const weights.
static inline float readInput(const float* x, int j) {
  return pgm_read_float(x + j);
}

#ifdef SPARSE_WEIGHTS

static inline uint16_t readRowPtr(const uint16_t* rowPtr, int idx) {
  return pgm_read_word(&rowPtr[idx]);
}

static inline uint16_t readColIdx(const uint16_t* colIdx, int idx) {
  return pgm_read_word(&colIdx[idx]);
}

static inline int8_t readWeightVal(const int8_t* weights, int idx) {
  return (int8_t)pgm_read_byte(&weights[idx]);
}

float Int8ValueNet::evaluate(const float* x) const {
  g_evalCalls++;

  float* h1 = s_h1;
  for (int i = 0; i < FC1_OUT_DIM; i++) {
    int32_t acc = 0;
    const uint16_t rowStart = readRowPtr(fc1_w_row_ptr, i);
    const uint16_t rowEnd = readRowPtr(fc1_w_row_ptr, i + 1);
    for (uint16_t k = rowStart; k < rowEnd; k++) {
      const int j = readColIdx(fc1_w_col_idx, k);
      if (readInput(x, j) > 0.5f) {
        acc += (int32_t)readWeightVal(fc1_w_val, k);
      }
    }
    float sum = (float)acc * fc1_w_scale * input_scale;
    int8_t bq = (int8_t)pgm_read_byte(&fc1_b[i]);
    sum += (float)bq * fc1_b_scale;
    h1[i] = (sum > 0.0f) ? sum : 0.0f;
  }

  float* h2 = s_h2;
  for (int i = 0; i < FC2_OUT_DIM; i++) {
    float sum = (float)(int8_t)pgm_read_byte(&fc2_b[i]) * fc2_b_scale;
    const uint16_t rowStart = readRowPtr(fc2_w_row_ptr, i);
    const uint16_t rowEnd = readRowPtr(fc2_w_row_ptr, i + 1);
    for (uint16_t k = rowStart; k < rowEnd; k++) {
      const int j = readColIdx(fc2_w_col_idx, k);
      const float act = h1[j];
      if (act > 0.0f) {
        const float w = (float)readWeightVal(fc2_w_val, k) * fc2_w_scale;
        sum += w * act;
      }
    }
    h2[i] = (sum > 0.0f) ? sum : 0.0f;
  }

  float out = (float)(int8_t)pgm_read_byte(&fc3_b[0]) * fc3_b_scale;
  const uint16_t rowStart = readRowPtr(fc3_w_row_ptr, 0);
  const uint16_t rowEnd = readRowPtr(fc3_w_row_ptr, 1);
  for (uint16_t k = rowStart; k < rowEnd; k++) {
    const int j = readColIdx(fc3_w_col_idx, k);
    const float act = h2[j];
    if (act > 0.0f) {
      const float w = (float)readWeightVal(fc3_w_val, k) * fc3_w_scale;
      out += w * act;
    }
  }
  asm volatile("" ::: "memory");
  return tanhf(out);
}

int8_t Int8ValueNet::weightAt(int idx) {
  if (idx < FC1_W_NNZ) {
    return readWeightVal(fc1_w_val, idx);
  }
  return 0;
}

#else  // dense weights

float Int8ValueNet::evaluate(const float* x) const {
  g_evalCalls++;

  float* h1 = s_h1;
  for (int i = 0; i < FC1_OUT_DIM; i++) {
    int32_t acc = 0;
    for (int j = 0; j < FC1_IN_DIM; j++) {
      if (readInput(x, j) > 0.5f) {
        const uint32_t idx = (uint32_t)i * (uint32_t)FC1_IN_DIM + (uint32_t)j;
        acc += (int32_t)(int8_t)pgm_read_byte(&fc1_w[idx]);
      }
    }
    float sum = (float)acc * fc1_w_scale * input_scale;
    int8_t bq = pgm_read_byte(&fc1_b[i]);
    sum += (float)bq * fc1_b_scale;
    h1[i] = (sum > 0.0f) ? sum : 0.0f;
  }

  float* h2 = s_h2;
  for (int i = 0; i < FC2_OUT_DIM; i++) {
    float sum = (float)(int8_t)pgm_read_byte(&fc2_b[i]) * fc2_b_scale;
    for (int j = 0; j < FC2_IN_DIM; j++) {
      if (h1[j] > 0.0f) {
        const uint32_t idx = (uint32_t)i * (uint32_t)FC2_IN_DIM + (uint32_t)j;
        const float w = (float)(int8_t)pgm_read_byte(&fc2_w[idx]) * fc2_w_scale;
        sum += w * h1[j];
      }
    }
    h2[i] = (sum > 0.0f) ? sum : 0.0f;
  }

  float out = (float)(int8_t)pgm_read_byte(&fc3_b[0]) * fc3_b_scale;
  for (int j = 0; j < FC3_IN_DIM; j++) {
    if (h2[j] > 0.0f) {
      const float w = (float)(int8_t)pgm_read_byte(&fc3_w[j]) * fc3_w_scale;
      out += w * h2[j];
    }
  }
  asm volatile("" ::: "memory");
  return tanhf(out);
}

int8_t Int8ValueNet::weightAt(int idx) {
  return (int8_t)pgm_read_byte(&fc1_w[idx]);
}

#endif  // SPARSE_WEIGHTS

void Int8ValueNet::printArchitecture() {
  Serial.print(FC1_IN_DIM); Serial.print("→"); Serial.print(FC1_OUT_DIM);
  Serial.print("→"); Serial.print(FC2_OUT_DIM);
  Serial.print("→"); Serial.println(FC3_OUT_DIM);
#ifdef SPARSE_WEIGHTS
  Serial.print("Sparse CSR weights: ");
  Serial.print(FC1_W_NNZ); Serial.print("+");
  Serial.print(FC2_W_NNZ); Serial.print("+");
  Serial.println(FC3_W_NNZ);
#endif
}