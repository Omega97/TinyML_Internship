#include "Benchmark.h"
#include "Int8ValueNet.h"
#include "WioBoard.h"

extern volatile uint32_t g_evalCalls;

void Benchmark::begin() {
  evals_ = 0;
  evalsAtLastReport_ = 0;
  lastReport_ = 0;
  totalEvalMicros_ = 0;
  totalEvalMicrosAtLastReport_ = 0;
  smoothedEvalsPerSec_ = 0.0f;
  smoothedLatencyMs_ = 0.0f;
  hasSmoothed_ = false;
  skipWarmupInterval_ = true;
  forwardSink_ = 0.0f;
}

void Benchmark::tick(Int8ValueNet& net, const float* input) {
  const uint32_t callsBefore = g_evalCalls;
  unsigned long t0 = micros();
  forwardSink_ = net.evaluate(input);
  totalEvalMicros_ += micros() - t0;
  evals_++;
  // If this fires, the compiler elided evaluate() — inference did not run.
  if (g_evalCalls == callsBefore) {
    forwardSink_ = 0.0f / forwardSink_;
  }
  asm volatile("" : "+r"(forwardSink_) :: "memory");
}

void Benchmark::reportIfDue(WioBoard& board) {
  unsigned long now = millis();

  if (!lastReport_) {
    lastReport_ = now;
    evalsAtLastReport_ = 0;
  }

  if (now - lastReport_ < 1000) {
    return;
  }

  unsigned long dtMs = now - lastReport_;
  unsigned long deltaEvals = evals_ - evalsAtLastReport_;
  unsigned long deltaMicros = totalEvalMicros_ - totalEvalMicrosAtLastReport_;
  float instantRate = (dtMs > 0)
    ? (1000.0f * (float)deltaEvals / (float)dtMs)
    : 0.0f;
  float instantLatencyMs = (deltaEvals > 0)
    ? ((float)deltaMicros / (float)deltaEvals) / 1000.0f
    : 0.0f;

  if (skipWarmupInterval_) {
    smoothedEvalsPerSec_ = instantRate;
    smoothedLatencyMs_ = instantLatencyMs;
    hasSmoothed_ = true;
    skipWarmupInterval_ = false;
  } else {
    if (!hasSmoothed_) {
      smoothedEvalsPerSec_ = instantRate;
      smoothedLatencyMs_ = instantLatencyMs;
      hasSmoothed_ = true;
    } else {
      smoothedEvalsPerSec_ = kEmaDecay * smoothedEvalsPerSec_
                           + (1.0f - kEmaDecay) * instantRate;
      smoothedLatencyMs_ = kEmaDecay * smoothedLatencyMs_
                         + (1.0f - kEmaDecay) * instantLatencyMs;
    }
    board.showBenchmarkStats(smoothedEvalsPerSec_, smoothedLatencyMs_);
  }

  evalsAtLastReport_ = evals_;
  totalEvalMicrosAtLastReport_ = totalEvalMicros_;
  lastReport_ = now;
}