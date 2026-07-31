import chess
import chess.engine

# Start the Cfish subprocess
engine = chess.engine.SimpleEngine.popen_uci("./cfish")

# Create a standard chess board
board = chess.Board()

# Evaluate position and get the best move
result = engine.play(board, chess.engine.Limit(time=1.0))
print(f"Best Move: {result.move}")

# Inspect evaluation score
info = engine.analyse(board, chess.engine.Limit(depth=12))
print(f"Score: {info['score'].relative}")

# Terminate process
engine.quit()
