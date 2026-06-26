#include "WioBoard.h"
#include "Int8ValueNet.h"

#include <TFT_eSPI.h>
#include <SPI.h>
#include <avr/pgmspace.h>

static TFT_eSPI tft = TFT_eSPI();

void WioBoard::begin() {
  Serial.begin(115200);
  unsigned long serialWait = millis();
  while (!Serial && millis() - serialWait < 3000) { }
  initDisplay();
}

void WioBoard::initDisplay() {
  tft.init();
  tft.setRotation(3);
  tft.fillScreen(TFT_BLACK);
  tft.setTextColor(TFT_WHITE);
}

void WioBoard::printBootInfo(const char* weightsFile, int8_t w0, int8_t w1, int8_t w2) {
  Serial.print("Loaded Weights File: ");
  Serial.println(weightsFile);
  Serial.print("Architecture: ");
  Int8ValueNet::printArchitecture();

  Serial.print("First 3 weights from fc1_w: ");
  Serial.print((int)w0);
  Serial.print(",");
  Serial.print((int)w1);
  Serial.print(",");
  Serial.println((int)w2);

  tft.setTextSize(1);
  tft.setCursor(10, 90);
  tft.print("Weights: ");
  tft.print(weightsFile);

  tft.setCursor(10, 110);
  tft.print("W[0:2]:");
  tft.print((int)w0);
  tft.print(",");
  tft.print((int)w1);
  tft.print(",");
  tft.print((int)w2);

  tft.setTextSize(2);
  tft.setCursor(10, 10);
  tft.print("Inferred value:");
}

void WioBoard::showInferredValue(float val) {
  Serial.print("Inferred value: ");
  Serial.println(val, 6);

  tft.setCursor(10, 30);
  tft.print(val, 6);
}

void WioBoard::showBenchmarkStats(float evalsPerSec, float latencyMs) {
  Serial.print("Avg Evals/s: ");
  Serial.print(evalsPerSec, 2);
  Serial.print("  Latency: ");
  Serial.print(latencyMs, 2);
  Serial.println(" ms");

  tft.fillRect(10, 60, 300, 32, TFT_BLACK);
  tft.setCursor(10, 60);
  tft.print("Evals/s: ");
  tft.println(evalsPerSec, 2);
  tft.setCursor(10, 76);
  tft.print("Latency: ");
  tft.print(latencyMs, 2);
  tft.println(" ms");
}