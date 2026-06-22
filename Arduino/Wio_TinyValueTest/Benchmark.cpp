#include "Benchmark.h"
#include "Int8ValueNet.h"
#include "WioBoard.h"

void Benchmark::begin() {
  evals_ = 0;
  evalsAtLastReport_ = 0;
  lastReport_ = 0;
  smoothedEvalsPerSec_ = 0.0f;
  hasSmoothed_ = false;
  skipWarmupInterval_ = true;
  forwardSink_ = 0.0f;
}

void Benchmark::tick(Int8ValueNet& net, const float* input) {
  forwardSink_ = net.evaluate(input);
  evals_++;
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
  float instantRate = (dtMs > 0)
    ? (1000.0f * (float)deltaEvals / (float)dtMs)
    : 0.0f;

  if (skipWarmupInterval_) {
    smoothedEvalsPerSec_ = instantRate;
    hasSmoothed_ = true;
    skipWarmupInterval_ = false;
  } else {
    if (!hasSmoothed_) {
      smoothedEvalsPerSec_ = instantRate;
      hasSmoothed_ = true;
    } else {
      smoothedEvalsPerSec_ = kEmaDecay * smoothedEvalsPerSec_
                           + (1.0f - kEmaDecay) * instantRate;
    }
    board.showEvalsPerSec(smoothedEvalsPerSec_);
  }

  evalsAtLastReport_ = evals_;
  lastReport_ = now;
}