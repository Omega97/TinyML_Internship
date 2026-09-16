/* SARDINE DualHidden WDL eval for Cfish. Full re-encode every call (no accumulators). */

#include "sardine_eval.h"

#include "bitboard.h"
#include "position.h"

#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SARDINE_FEAT 844
#define SARDINE_MAX_W 256
#define SARDINE_MAX_H 512
#define SARDINE_MAX_ACT 128
#define SARDINE_SCALE 400.0f
#define SARDINE_CLIP 127.0f

enum { S_WK = 1, S_WQ = 2, S_BK = 4, S_BQ = 8 };

static int16_t ps_index[2][6][64];
static int16_t king_self[8][4];
static int16_t king_enemy[8][8];
static int16_t castle_index[2][2];
static int16_t ep_index[8];
static int maps_ready;

static int feat_dim = SARDINE_FEAT, hidden_dim, hidden2_dim;
static float *l1_w, *l1_b, *l2_w, *l2_b, *head_w, *head_b;
static int weights_ok;
static _Atomic unsigned long long sardine_calls;

__attribute__((destructor))
static void sardine_stats(void)
{
  unsigned long long n = atomic_load(&sardine_calls);
  if (n)
    fprintf(stderr, "sardine: eval_calls=%llu\n", n);
}

static int sq_file(int s) { return s & 7; }
static int sq_rank(int s) { return s >> 3; }
static int flip_h(int s) { return s ^ 7; }
static int flip_v(int s) { return s ^ 56; }

static void build_maps(void)
{
  int idx = 0;
  int side, pt, sq, rank, mf, file;
  /* python: for side in (WHITE=True=1, BLACK=False=0) */
  for (side = 1; side >= 0; side--) {
    for (pt = 1; pt <= 5; pt++) {
      for (sq = 0; sq < 64; sq++) {
        rank = sq_rank(sq);
        if (pt == PAWN && (rank == 0 || rank == 7)) {
          ps_index[side][pt][sq] = -1;
          continue;
        }
        ps_index[side][pt][sq] = (int16_t)idx++;
      }
    }
  }
  for (rank = 0; rank < 8; rank++)
    for (mf = 0; mf < 4; mf++)
      king_self[rank][mf] = (int16_t)idx++;
  for (rank = 0; rank < 8; rank++)
    for (file = 0; file < 8; file++)
      king_enemy[rank][file] = (int16_t)idx++;
  castle_index[1][1] = (int16_t)(idx + 0); /* white kingside: META+0 */
  castle_index[1][0] = (int16_t)(idx + 1);
  castle_index[0][1] = (int16_t)(idx + 2);
  castle_index[0][0] = (int16_t)(idx + 3);
  for (file = 0; file < 8; file++)
    ep_index[file] = (int16_t)(idx + 4 + file);
  maps_ready = 1;
}

static int load_weights(const char *path)
{
  FILE *f;
  char magic[4];
  unsigned ver, feat, w, h;
  size_t n;
  const char *try_paths[4];
  int i;

  try_paths[0] = path;
  try_paths[1] = "sardine.bin";
  try_paths[2] = "engines/sardine.bin";
  try_paths[3] = "/home/omar/jupyterlab/lichess-bot/engines/sardine.bin";

  f = NULL;
  for (i = 0; i < 4; i++) {
    if (!try_paths[i])
      continue;
    f = fopen(try_paths[i], "rb");
    if (f)
      break;
  }
  if (!f) {
    fprintf(stderr, "sardine: cannot open sardine.bin\n");
    return 0;
  }
  if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "SRDN", 4) != 0) {
    fprintf(stderr, "sardine: bad magic\n");
    fclose(f);
    return 0;
  }
  if (fread(&ver, 4, 1, f) != 1 || fread(&feat, 4, 1, f) != 1
      || fread(&w, 4, 1, f) != 1 || fread(&h, 4, 1, f) != 1) {
    fclose(f);
    return 0;
  }
  if (feat != SARDINE_FEAT || w > SARDINE_MAX_W || h > SARDINE_MAX_H) {
    fprintf(stderr, "sardine: unsupported dims feat=%u W=%u H=%u\n", feat, w, h);
    fclose(f);
    return 0;
  }
  feat_dim = (int)feat;
  hidden_dim = (int)w;
  hidden2_dim = (int)h;
  n = (size_t)hidden_dim * (size_t)feat_dim;
  l1_w = malloc(n * sizeof(float));
  l1_b = malloc((size_t)hidden_dim * sizeof(float));
  l2_w = malloc((size_t)hidden2_dim * 2 * (size_t)hidden_dim * sizeof(float));
  l2_b = malloc((size_t)hidden2_dim * sizeof(float));
  head_w = malloc(3 * (size_t)hidden2_dim * sizeof(float));
  head_b = malloc(3 * sizeof(float));
  if (!l1_w || !l1_b || !l2_w || !l2_b || !head_w || !head_b) {
    fclose(f);
    return 0;
  }
  if (fread(l1_w, sizeof(float), n, f) != n
      || fread(l1_b, sizeof(float), (size_t)hidden_dim, f) != (size_t)hidden_dim
      || fread(l2_w, sizeof(float), (size_t)hidden2_dim * 2 * (size_t)hidden_dim, f)
           != (size_t)hidden2_dim * 2 * (size_t)hidden_dim
      || fread(l2_b, sizeof(float), (size_t)hidden2_dim, f) != (size_t)hidden2_dim
      || fread(head_w, sizeof(float), 3 * (size_t)hidden2_dim, f) != 3 * (size_t)hidden2_dim
      || fread(head_b, sizeof(float), 3, f) != 3) {
    fprintf(stderr, "sardine: short read\n");
    fclose(f);
    return 0;
  }
  fclose(f);
  weights_ok = 1;
  fprintf(stderr, "sardine: loaded DualHidden W=%d H=%d\n", hidden_dim, hidden2_dim);
  return 1;
}

int sardine_init(const char *path)
{
  static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
  int ok;
  pthread_mutex_lock(&mu);
  if (!maps_ready)
    build_maps();
  if (!weights_ok)
    load_weights(path);
  ok = weights_ok;
  pthread_mutex_unlock(&mu);
  return ok;
}

typedef struct {
  uint8_t piece[64];
  int ep;
  int castle;
} SBoard;

static void from_pos(const Position *pos, SBoard *b)
{
  Square s;
  b->ep = (int)ep_square();
  if (b->ep == 0)
    b->ep = 64;
  b->castle = 0;
  if (can_castle_cr(WHITE_OO))
    b->castle |= S_WK;
  if (can_castle_cr(WHITE_OOO))
    b->castle |= S_WQ;
  if (can_castle_cr(BLACK_OO))
    b->castle |= S_BK;
  if (can_castle_cr(BLACK_OOO))
    b->castle |= S_BQ;
  for (s = 0; s < 64; s++)
    b->piece[s] = (uint8_t)piece_on(s);
}

static int cfish_white(uint8_t pc)
{
  return pc && !(pc & 8);
}

static int cfish_pt(uint8_t pc)
{
  return (int)(pc & 7);
}

static void apply_h(SBoard *b)
{
  uint8_t tmp[64];
  int s, c = 0;
  for (s = 0; s < 64; s++)
    tmp[flip_h(s)] = b->piece[s];
  memcpy(b->piece, tmp, 64);
  if (b->ep < 64)
    b->ep = flip_h(b->ep);
  if (b->castle & S_WK)
    c |= S_WQ;
  if (b->castle & S_WQ)
    c |= S_WK;
  if (b->castle & S_BK)
    c |= S_BQ;
  if (b->castle & S_BQ)
    c |= S_BK;
  b->castle = c;
}

static void apply_mirror(SBoard *b)
{
  uint8_t tmp[64];
  int s, c = 0;
  for (s = 0; s < 64; s++) {
    uint8_t pc = b->piece[s];
    if (pc)
      pc ^= 8;
    tmp[flip_v(s)] = pc;
  }
  memcpy(b->piece, tmp, 64);
  if (b->ep < 64)
    b->ep = flip_v(b->ep);
  if (b->castle & S_WK)
    c |= S_BK;
  if (b->castle & S_WQ)
    c |= S_BQ;
  if (b->castle & S_BK)
    c |= S_WK;
  if (b->castle & S_BQ)
    c |= S_WQ;
  b->castle = c;
}

static int find_king(const SBoard *b, int py_white)
{
  int s;
  for (s = 0; s < 64; s++) {
    uint8_t pc = b->piece[s];
    if (pc && cfish_pt(pc) == KING && cfish_white(pc) == py_white)
      return s;
  }
  return -1;
}

static uint64_t occ_bb(const SBoard *b)
{
  uint64_t o = 0;
  int s;
  for (s = 0; s < 64; s++)
    if (b->piece[s])
      o |= 1ull << s;
  return o;
}

static int on_board_delta(int sq, int df, int dr)
{
  int f = sq_file(sq) + df, r = sq_rank(sq) + dr;
  if (f < 0 || f > 7 || r < 0 || r > 7)
    return -1;
  return r * 8 + f;
}

static uint64_t slider_attacks(uint64_t occ, int sq, const int *df, const int *dr, int n)
{
  uint64_t a = 0;
  int i, s, k;
  for (i = 0; i < n; i++) {
    s = sq;
    for (k = 0; k < 8; k++) {
      s = on_board_delta(s, df[i], dr[i]);
      if (s < 0)
        break;
      a |= 1ull << s;
      if (occ & (1ull << s))
        break;
    }
  }
  return a;
}

static uint64_t piece_attacks(const SBoard *b, uint64_t occ, int sq)
{
  static const int ndf[] = {1, -1, 2, -2, 1, -1, 2, -2};
  static const int ndr[] = {2, 2, 1, 1, -2, -2, -1, -1};
  static const int bdf[] = {1, 1, -1, -1};
  static const int bdr[] = {1, -1, 1, -1};
  static const int rdf[] = {1, -1, 0, 0};
  static const int rdr[] = {0, 0, 1, -1};
  static const int kdf[] = {1, 1, 1, 0, 0, -1, -1, -1};
  static const int kdr[] = {1, 0, -1, 1, -1, 1, 0, -1};
  uint8_t pc = b->piece[sq];
  int pt, i, t, py_white;
  uint64_t a = 0;
  if (!pc)
    return 0;
  pt = cfish_pt(pc);
  py_white = cfish_white(pc);
  if (pt == PAWN) {
    int dir = py_white ? 1 : -1;
    t = on_board_delta(sq, 1, dir);
    if (t >= 0)
      a |= 1ull << t;
    t = on_board_delta(sq, -1, dir);
    if (t >= 0)
      a |= 1ull << t;
    return a;
  }
  if (pt == KNIGHT) {
    for (i = 0; i < 8; i++) {
      t = on_board_delta(sq, ndf[i], ndr[i]);
      if (t >= 0)
        a |= 1ull << t;
    }
    return a;
  }
  if (pt == KING) {
    for (i = 0; i < 8; i++) {
      t = on_board_delta(sq, kdf[i], kdr[i]);
      if (t >= 0)
        a |= 1ull << t;
    }
    return a;
  }
  if (pt == BISHOP || pt == QUEEN)
    a |= slider_attacks(occ, sq, bdf, bdr, 4);
  if (pt == ROOK || pt == QUEEN)
    a |= slider_attacks(occ, sq, rdf, rdr, 4);
  return a;
}

static int attacked_by(const SBoard *b, uint64_t occ, int sq, int py_white_attacker)
{
  int s;
  for (s = 0; s < 64; s++) {
    uint8_t pc = b->piece[s];
    if (!pc || cfish_white(pc) != py_white_attacker)
      continue;
    if (piece_attacks(b, occ, s) & (1ull << sq))
      return 1;
  }
  return 0;
}

static int encode_view(const SBoard *view, const SBoard *castle_src, int mirrored,
                      int py_persp, int *idx)
{
  int n = 0, s, king_sq, py_color, pt, id;
  uint64_t occ = occ_bb(view);
  king_sq = find_king(view, py_persp);

  for (s = 0; s < 64; s++) {
    uint8_t pc = view->piece[s];
    if (!pc)
      continue;
    py_color = cfish_white(pc);
    pt = cfish_pt(pc);
    if (pt == KING) {
      if (py_color == py_persp)
        id = king_self[sq_rank(s)][sq_file(s) < 4 ? sq_file(s) : 7 - sq_file(s)];
      else
        id = king_enemy[sq_rank(s)][sq_file(s)];
    } else {
      id = ps_index[py_color][pt][s];
      if (id < 0)
        continue;
    }
    if (n < SARDINE_MAX_ACT)
      idx[n++] = id;
  }

  /* castling from pre-mirror-horizontal source, labels swapped if mirrored */
  {
    int wk = mirrored ? 0 : 1, wq = mirrored ? 1 : 0;
    int bk = mirrored ? 0 : 1, bq = mirrored ? 1 : 0;
    if (castle_src->castle & S_WK)
      idx[n++] = castle_index[1][wk];
    if (castle_src->castle & S_WQ)
      idx[n++] = castle_index[1][wq];
    if (castle_src->castle & S_BK)
      idx[n++] = castle_index[0][bk];
    if (castle_src->castle & S_BQ)
      idx[n++] = castle_index[0][bq];
  }

  if (view->ep < 64)
    idx[n++] = ep_index[sq_file(view->ep)];

  for (s = 0; s < 64; s++) {
    uint8_t pc = view->piece[s];
    if (!pc)
      continue;
    py_color = cfish_white(pc);
    if (py_color == py_persp && attacked_by(view, occ, s, !py_persp))
      idx[n++] = 716 + s;
    if (py_color != py_persp && king_sq >= 0
        && (piece_attacks(view, occ, s) & (1ull << king_sq)))
      idx[n++] = 716 + 64 + s;
  }
  return n;
}

static int encode_perspective(SBoard src, int py_persp, int *idx)
{
  SBoard castle_src = src;
  int king, mirrored = 0;
  king = find_king(&src, py_persp);
  if (king >= 0 && sq_file(king) >= 4) {
    apply_h(&src);
    mirrored = 1;
  }
  return encode_view(&src, &castle_src, mirrored, py_persp, idx);
}

static void crelu(float *x, int n)
{
  int i;
  for (i = 0; i < n; i++) {
    if (x[i] < 0.0f)
      x[i] = 0.0f;
    else if (x[i] > SARDINE_CLIP)
      x[i] = SARDINE_CLIP;
  }
}

static void l1_sparse(const int *idx, int n, float *out)
{
  int h, k;
  for (h = 0; h < hidden_dim; h++)
    out[h] = l1_b[h];
  for (k = 0; k < n; k++) {
    int i = idx[k];
    if (i < 0 || i >= feat_dim)
      continue;
    for (h = 0; h < hidden_dim; h++)
      out[h] += l1_w[h * feat_dim + i];
  }
  crelu(out, hidden_dim);
}

static void l2_forward(const float *stm, const float *opp, float *out)
{
  int h, i, in = hidden_dim * 2;
  for (h = 0; h < hidden2_dim; h++) {
    float s = l2_b[h];
    const float *row = l2_w + h * in;
    for (i = 0; i < hidden_dim; i++)
      s += row[i] * stm[i];
    for (i = 0; i < hidden_dim; i++)
      s += row[hidden_dim + i] * opp[i];
    out[h] = s;
  }
  crelu(out, hidden2_dim);
}

static float stm_value(const float *h2)
{
  float z[3], m, e0, e1, e2, den;
  int i;
  for (i = 0; i < 3; i++) {
    const float *row = head_w + i * hidden2_dim;
    float s = head_b[i];
    int k;
    for (k = 0; k < hidden2_dim; k++)
      s += row[k] * h2[k];
    z[i] = s;
  }
  m = z[0];
  if (z[1] > m)
    m = z[1];
  if (z[2] > m)
    m = z[2];
  e0 = expf(z[0] - m);
  e1 = expf(z[1] - m);
  e2 = expf(z[2] - m);
  den = e0 + e1 + e2;
  return (e0 - e2) / den;
}

Value sardine_evaluate(const Position *pos)
{
  SBoard b, mir;
  int widx[SARDINE_MAX_ACT], bidx[SARDINE_MAX_ACT], nw, nb;
  float wh[SARDINE_MAX_W], bh[SARDINE_MAX_W], h2[SARDINE_MAX_H];
  const float *stm, *opp;
  float v;

  if (!sardine_init(NULL))
    return 0;
  atomic_fetch_add_explicit(&sardine_calls, 1, memory_order_relaxed);

  from_pos(pos, &b);
  nw = encode_perspective(b, 1, widx);
  mir = b;
  apply_mirror(&mir);
  nb = encode_perspective(mir, 1, bidx);

  l1_sparse(widx, nw, wh);
  l1_sparse(bidx, nb, bh);

  if (stm() == WHITE) {
    stm = wh;
    opp = bh;
  } else {
    stm = bh;
    opp = wh;
  }
  l2_forward(stm, opp, h2);
  v = stm_value(h2);
  {
    static int dumped;
    Value score = (Value)(v * SARDINE_SCALE + (v >= 0.0f ? 0.5f : -0.5f));
    if (!dumped) {
      dumped = 1;
      fprintf(stderr, "sardine: first eval stm=%.4f cp=%d nw=%d nb=%d\n", v, (int)score, nw, nb);
    }
    return score;
  }
}
