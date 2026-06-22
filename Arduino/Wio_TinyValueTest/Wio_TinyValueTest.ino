/*
 * TinyML Neural Network - Seeed Wio Terminal value-net stress test.
 *
 * Sketch layout:
 *   config.h            — WEIGHTS_FILE switch + weight / FEN headers
 *   Int8ValueNet.h/.cpp — int8 MLP forward pass (inline evaluate)
 *   WioBoard.h/.cpp     — Serial + TFT display
 *   Benchmark.h/.cpp    — honest evals/sec loop (volatile sink + interval EMA)
 *
 * Generated headers (run from project root):
 *   wio_int8_weights_nano.h   ←  py -3.12 scripts/prepare_wio_nano.py
 *   wio_int8_weights_tiny.h   ←  py -3.12 scripts/prepare_wio_tiny.py
 *   wio_intc:\Users\monfalcone\PycharmProjects\TinyMLInternship\Arduino\Wio_TinyValueTest\Wio_TinyValueTest.ino8_weights_small.h  ←  py -3.12 scripts/prepare_wio_small.py
 *   wio_int8_weights_medium.h ←  py -3.12 scripts/prepare_wio_medium.py
 *   wio_int8_weights_big.h    ←  py -3.12 scripts/prepare_wio_big.py
 *   wio_int8_weights_huge.h   ←  py -3.12 scripts/prepare_wio_huge.py
 *   fen_input.h               ←  py -3.12 scripts/fen_to_c_array.py "FEN" --output Arduino/Wio_TinyValueTest/fen_input.h
 *
 * Upload: Board = Seeed Wio Terminal, Port = your COM port, Serial = 115200.
 */

#include "config.h"
#include "Int8ValueNet.h"
#include "WioBoard.h"
#include "Benchmark.h"

Int8ValueNet net;
WioBoard board;
Benchmark bench;

void setup() {
  board.begin();
  bench.begin();

  board.printBootInfo(WEIGHTS_FILE,
                      Int8ValueNet::weightAt(0),
                      Int8ValueNet::weightAt(1),
                      Int8ValueNet::weightAt(2));

  float val = net.evaluate(input);
  board.showInferredValue(val);
}

void loop() {
  bench.tick(net, input);
  bench.reportIfDue(board);
}