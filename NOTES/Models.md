
## Dense/Convolutional Models, AlphaZero, Lc0

#### Marvin
- [Marvin](https://huggingface.co/holymolyyy/marvin) is a human-like chess neural network that models human play at a specified Elo rating and time control. 
- **Architecture:** _Not Found_ (Neural network trained to emulate human-like playstyles).
- **Input:** _Not Found_.
- **Output Description:** _Not Found_ (Likely move probabilities matching human play distributions at target ratings).
- **Model Sizes:** Available in three sizes: **large**, **small**, and **tiny**.
- **Parameters:** _Not Found_.
- **File Size:** ~19.3 MB (for the `tiny_2400.pb.gz` checkpoint variant).

#### chess-bot
- [chess-bot](https://huggingface.co/AubreeL/chess-bot) is a policy-value network inspired by AlphaZero, designed to evaluate chess positions and suggest moves. 
- **Input**: 18-plane board representation (12 pieces + 6 metadata planes) 
- **Convolutional backbone**: 32 filters, 1 residual block, ~9,611,202 parameters. 
- **Policy head**: 4,672-dimensional output (one per legal move encoding). 
- **Value head**: Single tanh output (-1 to +1 for position evaluation) 
- **File Size:** 38 MB

#### chessmate-net
- [chessmate-net](https://huggingface.co/victorqueiroz/chessmate-net)
- **Input:** single-position `8×8×22` planes (pieces ×12, castling, side-to-move, en-passant, halfmove, attack maps), side-to-move oriented (board rotated for Black). 
- **Outputs:** 
	- `policy` — 4096 logits over `from*64 + to` (the parity-locked move encoding; promotion folds onto the from→to index, no underpromotion dim); 
	- `value` — scalar `tanh` in `[-1, 1]`, side-to-move POV. 
- **File Size:** 120 MB

#### chess-bot / TinyPCN
- [chess-bot](https://huggingface.co/AubreeL/chess-bot)
- Architecture: Policy-value network ispirata ad AlphaZero.
- Policy + Value: Sì.
- Parameters: circa 9.6M.

#### chessmate-net
- [models](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/77983406/9fe7729c-db8d-4f27-b7d0-e93109e825c0/Models.md?AWSAccessKeyId=ASIA2F3EMEYEQTA4RJEK&Signature=2NYsCVeWN80moMz39609lQmhoMc%3D&x-amz-security-token=IQoJb3JpZ2luX2VjEOH%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJHMEUCIQD%2BCt64%2FQVPJxq1kxzwRhDxJWWwlHRJrylXZ2%2BhidK7GgIgTakO%2BrPNrK15t%2B%2BCGW0xtPbx9Rk1O4QATZNjVUg4hFIq%2FAQIqf%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARABGgw2OTk3NTMzMDk3MDUiDMG6sfk29Xa910jDiCrQBMZQMghdcxJr8GCeYtL72jjE0CiLYTzVkexstFxJrRabhuGlbVHldSBTK40Uej0Gw4T%2FXkTU0PfcZ089tJdx18RGp98DKNNGoQLJ%2F1HWDMztJCQd0syvwpC28l6b6%2BoAbqeHLqqhY3I4r2bS73QahDJ69chKWI1pFGeDnAsV23cE10LqHJcfvOzC9zjskfC7Wv7sIwJcEnUox6fSDP7nuVPc2RHPrxMWHpjGnLAFeVqOhfbAnDZCvxnoLGfu38u%2B2WMi2%2B6Kak6nZCVWpTTcO7Kzuvu0aBpN3SWD%2BXIvwxGFT4xU9bamIinHv2NC163kEzmFKKOAp7Sp5Bl%2FmA3JwRGgnFNojci6ZgiQ%2Bqx5XKU3emlWWZ5h2eSUsA5bGuhWt2IG3iTe49tXI0wYtPMi6dC7uR8ZMfRx%2F6mE4Yz9oxnrkInF6sdS98bJN%2BXawrXKV%2FJEwSOTPwaS9REiZzUjXDxZ3jcbvM7kKz7HlvJQW8uRJAO5wtQa4REgRR1v82MniFlacRa%2FtDZtq5kVQDjdgMS1pT%2FzkUIKPpyP98rJWKLs%2FPSZxgpYGdTtymG5UNDnQaXg8LYry9Xc%2Fm4ye3OEFadNhexfrya%2BpznZCJbSVeTT%2Fb5TGX6WfHWTI7ACrp1kRMTKTjnCP42ZOlyA4Y2DYjw7ZYFTpQL6Rs3UhZcHPmjALH8cB9qR6q9uoaVr2R%2FEc701fnZak7rwTgH2YmRmdU%2BFa0X7tvMiTpvNnG%2F9KMyKGKIbt5zzK6DGdoofheBnf%2FNrNIZxkNEVmVGoglG6eaEw2tSI0gY6mAHfZxswa6e5nbJUl%2Buo0JFJ%2B2mblcanASHLKSH1HNQHBRQDdF3AU40zpiBRh45QQWKv2F2OO6HwkzGcKGuBe2TsbfOcqOOnwPdc6y8ZrEfpAWeiN9aXSNj2d1Ilq1Ym1ODdIfcGdFOValbgohIMjmHmpi9%2FmwOV6YpqX0YFTO4OMrucX18Vh7qpcFsyCJSLyXRVz3YyeE2LBw%3D%3D&Expires=1782724653)
- Architecture: Dense / fully-connected-style, secondo la nota disponibile.
- Policy + Value: Sì.
- Parameters: non indicati.


- Small Lc0 (Leela Chess Zero) Networks on Edge Devices. While the main Leela Chess Zero (Lc0) project relies on massive GPU clusters, the community has developed "personality nets" and smaller network architectures that can run on devices like the Raspberry Pi.
    - **[The Ultimate Guide to Lc0's Human-Like Personality Nets (Google Groups)](https://groups.google.com/g/picochess/c/ap95BZ2JIPg)**: Discusses how smaller Lc0 architectures can run on a Raspberry Pi 3, hitting around ~2400 Lichess Elo with just 8 nodes per move.
    - **[Autonomous chess-playing robotic arm using Raspberry PI (Scribd)](https://www.scribd.com/document/901155249/Autonomous-chess-playing-robotic-arm-using-Raspberry-PI?spm=a2ty_o01.29997173.0.0.267155fbh2DRlp)**: Demonstrates a practical implementation of running a chess engine (Stockfish) on a Raspberry Pi 3B+ paired with a camera for vision processing.
    
- https://github.com/LeelaChessZero/lc0 Lc0 on GitHub 
    
- https://lczero.org/ The Leela Chess Zero community has trained thousands of networks, including highly stripped-down architectures meant for testing or ultra-low-power CPU inference. Because they use a standardized network structure, you can parse them directly using a simple custom script or find community-converted ONNX files

---

## Res-Net

#### chess-alphazero-openenv
- [chess-alphazero-openenv](https://huggingface.co/yogeshwaran13/chess-alphazero-openenv) 
- **Architecture:** ResNet backbone featuring 5 residual blocks and 64 channels.
- **Input** : 19 × 8 × 8 board tensor 
- **Output Description:** Dual-headed output providing a: 
	- Move Policy (4096 moves distribution) 
	- Position Value (scalar from -1 to +1)
- **File Size:** 18.4 MB

#### chess-alphazero-openenv
- [Model](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/77983406/9fe7729c-db8d-4f27-b7d0-e93109e825c0/Models.md?AWSAccessKeyId=ASIA2F3EMEYEQTA4RJEK&Signature=2NYsCVeWN80moMz39609lQmhoMc%3D&x-amz-security-token=IQoJb3JpZ2luX2VjEOH%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJHMEUCIQD%2BCt64%2FQVPJxq1kxzwRhDxJWWwlHRJrylXZ2%2BhidK7GgIgTakO%2BrPNrK15t%2B%2BCGW0xtPbx9Rk1O4QATZNjVUg4hFIq%2FAQIqf%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARABGgw2OTk3NTMzMDk3MDUiDMG6sfk29Xa910jDiCrQBMZQMghdcxJr8GCeYtL72jjE0CiLYTzVkexstFxJrRabhuGlbVHldSBTK40Uej0Gw4T%2FXkTU0PfcZ089tJdx18RGp98DKNNGoQLJ%2F1HWDMztJCQd0syvwpC28l6b6%2BoAbqeHLqqhY3I4r2bS73QahDJ69chKWI1pFGeDnAsV23cE10LqHJcfvOzC9zjskfC7Wv7sIwJcEnUox6fSDP7nuVPc2RHPrxMWHpjGnLAFeVqOhfbAnDZCvxnoLGfu38u%2B2WMi2%2B6Kak6nZCVWpTTcO7Kzuvu0aBpN3SWD%2BXIvwxGFT4xU9bamIinHv2NC163kEzmFKKOAp7Sp5Bl%2FmA3JwRGgnFNojci6ZgiQ%2Bqx5XKU3emlWWZ5h2eSUsA5bGuhWt2IG3iTe49tXI0wYtPMi6dC7uR8ZMfRx%2F6mE4Yz9oxnrkInF6sdS98bJN%2BXawrXKV%2FJEwSOTPwaS9REiZzUjXDxZ3jcbvM7kKz7HlvJQW8uRJAO5wtQa4REgRR1v82MniFlacRa%2FtDZtq5kVQDjdgMS1pT%2FzkUIKPpyP98rJWKLs%2FPSZxgpYGdTtymG5UNDnQaXg8LYry9Xc%2Fm4ye3OEFadNhexfrya%2BpznZCJbSVeTT%2Fb5TGX6WfHWTI7ACrp1kRMTKTjnCP42ZOlyA4Y2DYjw7ZYFTpQL6Rs3UhZcHPmjALH8cB9qR6q9uoaVr2R%2FEc701fnZak7rwTgH2YmRmdU%2BFa0X7tvMiTpvNnG%2F9KMyKGKIbt5zzK6DGdoofheBnf%2FNrNIZxkNEVmVGoglG6eaEw2tSI0gY6mAHfZxswa6e5nbJUl%2Buo0JFJ%2B2mblcanASHLKSH1HNQHBRQDdF3AU40zpiBRh45QQWKv2F2OO6HwkzGcKGuBe2TsbfOcqOOnwPdc6y8ZrEfpAWeiN9aXSNj2d1Ilq1Ym1ODdIfcGdFOValbgohIMjmHmpi9%2FmwOV6YpqX0YFTO4OMrucX18Vh7qpcFsyCJSLyXRVz3YyeE2LBw%3D%3D&Expires=1782724653)
- Architecture: ResNet.
- Policy + Value: Sì.
- Parameters: non indicati.

#### chess-alphazero-pytorch
- [chess-alphazero-pytorch](https://huggingface.co/santoshchandu/chess-alphazero-pytorch) full tree-search
- Residual CNN (6 blocks, 128 filters, 10M parameters)
- Policy head: move probability over 4096 possible moves
- Value head: win probability in [-1, +1]
- MCTS: 100 simulations per move
- Size: ~41MB

---

## Transformers

#### pawn-small
- [pawn-small](https://huggingface.co/thomas-schweich/pawn-small) **PAWN** (Playstyle-Agnostic World-model Network for Chess) 
- a causal transformer trained on random chess games. 
- It learns legal moves, board state representations, and game dynamics purely from uniformly random legal move sequences -- no strategic play, no hand-crafted features, no external game databases. 
- **Architecture**: Transformer
- **File Size:** ~8.9M

#### ChessFormer-SL
- [ChessFormer-SL](https://huggingface.co/kaupane/ChessFormer-SL)
- Architecture: Transformer.
- Policy + Value: Sì, policy su 1.969 mosse e value head.
- Parameters: 100.7M.


 **ChessBot**
- [ChessBot](https://huggingface.co/Maxlegrec/ChessBot)
- Platform: Hugging Face.
- Architecture: Transformer.
- Policy + Value: Sì, policy e value.
- Parameters: 34.7M.

---

## NNUE

Originally invented by Yu Nasu in 2018 for Shogi and ported to computer chess in 2020 via Stockfish, NNUE completely changed the paradigm of embedded board-game AI.

- [**NNUE**](https://beuke.org/nnue/#:~:text=Therefore%2C%20instead%20of%20recomputing%20the,move%20is%20made%20or%20unmade.) is the undisputed gold standard for high-performance chess on low-resource hardware. Originally developed for Shogi and now famously integrated into the Stockfish engine, NNUE is specifically designed to run efficiently on standard CPUs without requiring a GPU:
    - **[Introducing NNUE Evaluation - Stockfish Blog](https://stockfishchess.org/blog/2020/introducing-nnue-evaluation/)**: The official announcement detailing how NNUE achieves low-latency evaluations on CPUs by only updating parts of the network incrementally
    - **[Efficiently Updatable Neural Network - Grokipedia](https://grokipedia.com/page/Efficiently_updatable_neural_network)**: Breaks down the architecture, noting the overparameterized input layer and the lightweight, updatable intermediate representation
    - **[Why Stockfish is So Good (Dev.to)](https://dev.to/djinn/why-stockfish-is-so-good-and-how-you-could-write-a-chess-engine-2lck)**: A technical deep dive into how NNUE uses int8/int16 quantization and SIMD to achieve massive speeds on consumer CPUs
    
    
- [beuke.org](https://beuke.org/nnue/#:~:text=Therefore%2C%20instead%20of%20recomputing%20the,move%20is%20made%20or%20unmade.) If you are looking for the absolute cutting edge of "small hardware" chess architectures, you must look at the FIDE & Google Efficient Chess AI Challenge hosted on Kaggle. This competition explicitly forbids brute-force computation and forces participants to build engines under extreme hardware constraints, such as allocating a maximum of just 5MiB of RAM.

---

### JEPA

[Yann LeCun, June 2022](https://openreview.net/pdf?id=BZ5a1r-kVsf)


---

## Other Pages
   
- The absolute cutting edge of "small hardware" chess architectures, the FIDE & Google Efficient Chess AI Challenge hosted on [Kaggle](https://www.kaggle.com/competitions/fide-google-efficiency-chess-ai-challenge/overview). This competition explicitly forbids brute-force computation and forces participants to build engines under extreme hardware constraints, such as allocating a maximum of just 5MiB of RAM
	- **[FIDE and Google create the Efficient Chess AI Challenge (FIDE.com)](https://www.fide.com/fide-and-google-create-the-efficient-chess-ai-challenge-hosted-on-kaggle/)**: The official announcement challenging developers to create smart, resource-light chess programs
	- **[FIDE & Google Efficient Chess AI Challenge (Kaggle)](https://www.kaggle.com/competitions/fide-google-efficiency-chess-ai-challenge/discussion/557921)**: The competition forums and discussion boards are a goldmine for seeing the exact lightweight architectures (like heavily pruned NNUE variants or tiny custom networks) that top competitors used to maximize Elo per byte
    
- https://huggingface.co/spaces/karthickajan/chess PHOTO -> FEN
    
- https://huggingface.co/atamano/whisper-chess-tiny reading move notation
    
- https://github.com/undera/chess-engine-nn chess-engine-nn
    

---
