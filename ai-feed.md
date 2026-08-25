
```powershell
C:\Users\monfalcone\PycharmProjects\TinyMLInternship>py -3.12 -u scripts/train_linear_wdl.py --epochs 100 --smoke --run-name linear_wdl_smoke --plot plots/linear_wdl_smoke_ce.png
Train: all slices in C:\Users\monfalcone\PycharmProjects\TinyMLInternship\data\processed\board_eval\fen_value_visits except fen_value_visits_lichess_db_standard_rated_2026-07_0-5000_d90
Test:  C:\Users\monfalcone\PycharmProjects\TinyMLInternship\data\processed\board_eval\fen_value_visits\fen_value_visits_lichess_db_standard_rated_2026-07_0-5000_d90
  train pool 904,717 in 31 slices | test 31,378 | batch 2048 mixed across slices × 9 steps/epoch
linear 844×2 → 3 WDL softmax | 5,067 params | loss=soft CE | train steps 9×2048 | test 31,378 | cpu
Output: C:\Users\monfalcone\PycharmProjects\TinyMLInternship\models\checkpoints\nnue\linear_wdl_smoke
epoch 01 | train_ce=1.086364 | test_ce=1.068795 | test_mae=0.574687 | 2.5s
epoch 02 | train_ce=1.059972 | test_ce=1.048391 | test_mae=0.562985 | 2.6s
epoch 03 | train_ce=1.043207 | test_ce=1.033606 | test_mae=0.551684 | 2.6s
epoch 04 | train_ce=1.032248 | test_ce=1.021907 | test_mae=0.542098 | 2.8s
epoch 05 | train_ce=1.022271 | test_ce=1.013132 | test_mae=0.534142 | 3.0s
epoch 06 | train_ce=1.013979 | test_ce=1.005549 | test_mae=0.526822 | 2.9s
epoch 07 | train_ce=1.007896 | test_ce=0.999030 | test_mae=0.520162 | 3.0s
epoch 08 | train_ce=1.002174 | test_ce=0.993486 | test_mae=0.514375 | 3.0s
epoch 09 | train_ce=0.997040 | test_ce=0.988331 | test_mae=0.509470 | 3.0s
epoch 10 | train_ce=0.992553 | test_ce=0.983675 | test_mae=0.504153 | 2.9s
epoch 11 | train_ce=0.985331 | test_ce=0.979578 | test_mae=0.499843 | 3.0s
epoch 12 | train_ce=0.980575 | test_ce=0.975469 | test_mae=0.495391 | 3.0s
epoch 13 | train_ce=0.979591 | test_ce=0.971963 | test_mae=0.491490 | 3.0s
epoch 14 | train_ce=0.976143 | test_ce=0.968467 | test_mae=0.487522 | 3.0s
epoch 15 | train_ce=0.973764 | test_ce=0.965365 | test_mae=0.484576 | 2.9s
epoch 16 | train_ce=0.970855 | test_ce=0.962425 | test_mae=0.480785 | 2.9s
epoch 17 | train_ce=0.969717 | test_ce=0.959805 | test_mae=0.478034 | 2.8s
epoch 18 | train_ce=0.963811 | test_ce=0.957232 | test_mae=0.474863 | 3.0s
epoch 19 | train_ce=0.961034 | test_ce=0.954752 | test_mae=0.472634 | 2.9s
epoch 20 | train_ce=0.959717 | test_ce=0.952749 | test_mae=0.469986 | 2.8s
epoch 21 | train_ce=0.953537 | test_ce=0.950099 | test_mae=0.467256 | 3.0s
epoch 22 | train_ce=0.952646 | test_ce=0.947830 | test_mae=0.464437 | 3.0s
epoch 23 | train_ce=0.953087 | test_ce=0.945706 | test_mae=0.462084 | 3.0s
epoch 24 | train_ce=0.952067 | test_ce=0.943678 | test_mae=0.459691 | 3.0s
epoch 25 | train_ce=0.949777 | test_ce=0.941737 | test_mae=0.457787 | 3.0s
epoch 26 | train_ce=0.951281 | test_ce=0.940030 | test_mae=0.456060 | 3.0s
epoch 27 | train_ce=0.946513 | test_ce=0.938463 | test_mae=0.454218 | 3.0s
epoch 28 | train_ce=0.947639 | test_ce=0.936593 | test_mae=0.452360 | 3.0s
epoch 29 | train_ce=0.943264 | test_ce=0.934921 | test_mae=0.450521 | 3.0s
epoch 30 | train_ce=0.942519 | test_ce=0.933224 | test_mae=0.448349 | 3.1s
epoch 31 | train_ce=0.939422 | test_ce=0.931641 | test_mae=0.446540 | 3.0s
epoch 32 | train_ce=0.939473 | test_ce=0.929981 | test_mae=0.444664 | 3.0s
epoch 33 | train_ce=0.939076 | test_ce=0.928534 | test_mae=0.443398 | 3.1s
epoch 34 | train_ce=0.934178 | test_ce=0.926978 | test_mae=0.441522 | 2.9s
epoch 35 | train_ce=0.931726 | test_ce=0.925575 | test_mae=0.439996 | 3.0s
epoch 36 | train_ce=0.928782 | test_ce=0.924070 | test_mae=0.437959 | 3.0s
epoch 37 | train_ce=0.932138 | test_ce=0.922653 | test_mae=0.436223 | 3.0s
epoch 38 | train_ce=0.929576 | test_ce=0.921772 | test_mae=0.435563 | 2.9s
epoch 39 | train_ce=0.927180 | test_ce=0.920107 | test_mae=0.434082 | 3.1s
epoch 40 | train_ce=0.923024 | test_ce=0.918775 | test_mae=0.431919 | 3.1s
epoch 41 | train_ce=0.926677 | test_ce=0.917641 | test_mae=0.431123 | 3.0s
epoch 42 | train_ce=0.924229 | test_ce=0.916561 | test_mae=0.429825 | 3.0s
epoch 43 | train_ce=0.917543 | test_ce=0.915098 | test_mae=0.428232 | 2.9s
epoch 44 | train_ce=0.923913 | test_ce=0.913931 | test_mae=0.426591 | 3.1s
epoch 45 | train_ce=0.919432 | test_ce=0.912852 | test_mae=0.425626 | 2.9s
epoch 46 | train_ce=0.920403 | test_ce=0.911614 | test_mae=0.424282 | 3.0s
epoch 47 | train_ce=0.916354 | test_ce=0.910676 | test_mae=0.423227 | 3.0s
epoch 48 | train_ce=0.921454 | test_ce=0.909603 | test_mae=0.422509 | 3.0s
epoch 49 | train_ce=0.913142 | test_ce=0.908779 | test_mae=0.421241 | 2.9s
epoch 50 | train_ce=0.916055 | test_ce=0.907449 | test_mae=0.419897 | 2.9s
epoch 51 | train_ce=0.911446 | test_ce=0.906315 | test_mae=0.418443 | 3.1s
epoch 52 | train_ce=0.912099 | test_ce=0.905315 | test_mae=0.417016 | 2.9s
epoch 53 | train_ce=0.908737 | test_ce=0.904485 | test_mae=0.416584 | 3.0s
epoch 54 | train_ce=0.911216 | test_ce=0.903521 | test_mae=0.415183 | 3.0s
epoch 55 | train_ce=0.911546 | test_ce=0.902565 | test_mae=0.414772 | 2.9s
epoch 56 | train_ce=0.913305 | test_ce=0.901574 | test_mae=0.413314 | 2.9s
epoch 57 | train_ce=0.908718 | test_ce=0.900610 | test_mae=0.412504 | 2.9s
epoch 58 | train_ce=0.909346 | test_ce=0.899529 | test_mae=0.411284 | 2.9s
epoch 59 | train_ce=0.905929 | test_ce=0.898563 | test_mae=0.410328 | 2.9s
epoch 60 | train_ce=0.905990 | test_ce=0.897576 | test_mae=0.408844 | 3.0s
epoch 61 | train_ce=0.901786 | test_ce=0.896550 | test_mae=0.408081 | 3.1s
epoch 62 | train_ce=0.900008 | test_ce=0.895679 | test_mae=0.407254 | 3.0s
epoch 63 | train_ce=0.902513 | test_ce=0.894797 | test_mae=0.406386 | 3.1s
epoch 64 | train_ce=0.900963 | test_ce=0.893932 | test_mae=0.405053 | 2.9s
epoch 65 | train_ce=0.898319 | test_ce=0.893161 | test_mae=0.404300 | 3.1s
epoch 66 | train_ce=0.893915 | test_ce=0.892468 | test_mae=0.403635 | 3.0s
epoch 67 | train_ce=0.896528 | test_ce=0.891715 | test_mae=0.402766 | 3.1s
epoch 68 | train_ce=0.901410 | test_ce=0.891189 | test_mae=0.402251 | 3.0s
epoch 69 | train_ce=0.896374 | test_ce=0.890271 | test_mae=0.401369 | 2.9s
epoch 70 | train_ce=0.896793 | test_ce=0.889403 | test_mae=0.399815 | 3.1s
epoch 71 | train_ce=0.898797 | test_ce=0.888723 | test_mae=0.399532 | 3.0s
epoch 72 | train_ce=0.897644 | test_ce=0.887930 | test_mae=0.398494 | 3.0s
epoch 73 | train_ce=0.890971 | test_ce=0.887205 | test_mae=0.398459 | 3.0s
epoch 74 | train_ce=0.893862 | test_ce=0.886351 | test_mae=0.396670 | 3.1s
epoch 75 | train_ce=0.893796 | test_ce=0.885585 | test_mae=0.396295 | 3.0s
epoch 76 | train_ce=0.894026 | test_ce=0.884918 | test_mae=0.395733 | 3.0s
epoch 77 | train_ce=0.888005 | test_ce=0.884129 | test_mae=0.394337 | 3.0s
epoch 78 | train_ce=0.889249 | test_ce=0.883339 | test_mae=0.393568 | 2.9s
epoch 79 | train_ce=0.893268 | test_ce=0.882762 | test_mae=0.393163 | 3.0s
epoch 80 | train_ce=0.891998 | test_ce=0.882109 | test_mae=0.392136 | 3.0s
epoch 81 | train_ce=0.897407 | test_ce=0.881680 | test_mae=0.392339 | 3.0s
epoch 82 | train_ce=0.888002 | test_ce=0.880846 | test_mae=0.391236 | 3.0s
epoch 83 | train_ce=0.888000 | test_ce=0.880147 | test_mae=0.390120 | 2.9s
epoch 84 | train_ce=0.890269 | test_ce=0.879653 | test_mae=0.390246 | 3.0s
epoch 85 | train_ce=0.886293 | test_ce=0.878949 | test_mae=0.388999 | 3.0s
epoch 86 | train_ce=0.887651 | test_ce=0.878235 | test_mae=0.388335 | 3.0s
epoch 87 | train_ce=0.883674 | test_ce=0.877639 | test_mae=0.387784 | 3.0s
epoch 88 | train_ce=0.877943 | test_ce=0.876983 | test_mae=0.387126 | 3.0s
epoch 89 | train_ce=0.876874 | test_ce=0.876359 | test_mae=0.385295 | 3.0s
epoch 90 | train_ce=0.880703 | test_ce=0.875671 | test_mae=0.385444 | 3.0s
epoch 91 | train_ce=0.881076 | test_ce=0.875116 | test_mae=0.385047 | 3.1s
epoch 92 | train_ce=0.872279 | test_ce=0.874543 | test_mae=0.384131 | 2.9s
epoch 93 | train_ce=0.878270 | test_ce=0.874002 | test_mae=0.383318 | 3.0s
epoch 94 | train_ce=0.876069 | test_ce=0.873421 | test_mae=0.382740 | 3.0s
epoch 95 | train_ce=0.881196 | test_ce=0.872997 | test_mae=0.382376 | 3.0s
epoch 96 | train_ce=0.883713 | test_ce=0.872665 | test_mae=0.382615 | 3.0s
epoch 97 | train_ce=0.875676 | test_ce=0.871898 | test_mae=0.381403 | 2.9s
epoch 98 | train_ce=0.877256 | test_ce=0.871262 | test_mae=0.380572 | 3.1s
epoch 99 | train_ce=0.878988 | test_ce=0.870763 | test_mae=0.380068 | 2.9s
epoch 100 | train_ce=0.878923 | test_ce=0.870304 | test_mae=0.379916 | 3.0s
Best test_ce=0.870304 → C:\Users\monfalcone\PycharmProjects\TinyMLInternship\models\checkpoints\nnue\linear_wdl_smoke\best.pt
CE plot → C:\Users\monfalcone\PycharmProjects\TinyMLInternship\plots\linear_wdl_smoke_ce.png
```