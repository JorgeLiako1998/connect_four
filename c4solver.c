/* Exact Connect Four solver (7x6): bitboard negamax + alpha-beta + transposition table.
 *
 * Score convention (from the point of view of the player to move):
 *    > 0  the mover wins  (bigger = faster win)
 *    = 0  draw
 *    < 0  the mover loses (more negative = faster loss)
 * Build:  clang -O3 -march=native -shared -fPIC c4solver.c -o libc4.dylib
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef uint64_t u64;
#define W 7
#define H 6
#define BOTTOM ((1ULL << 0) | (1ULL << 7) | (1ULL << 14) | (1ULL << 21) | (1ULL << 28) | (1ULL << 35) | (1ULL << 42))
#define BOARD (BOTTOM * ((1ULL << H) - 1))
#define COLMASK(c) (((1ULL << H) - 1) << ((c) * (H + 1)))

typedef struct { u64 cur, mask; int moves; } Pos;

static const int ORDER[W] = {3, 2, 4, 1, 5, 0, 6};

static u64 winning_cells(u64 p, u64 mask) {
    u64 r = (p << 1) & (p << 2) & (p << 3);
    u64 t = (p << (H + 1)) & (p << 2 * (H + 1));
    r |= t & (p << 3 * (H + 1)); r |= t & (p >> (H + 1));
    t = (p >> (H + 1)) & (p >> 2 * (H + 1));
    r |= t & (p << (H + 1)); r |= t & (p >> 3 * (H + 1));
    t = (p << H) & (p << 2 * H);
    r |= t & (p << 3 * H); r |= t & (p >> H);
    t = (p >> H) & (p >> 2 * H);
    r |= t & (p << H); r |= t & (p >> 3 * H);
    t = (p << (H + 2)) & (p << 2 * (H + 2));
    r |= t & (p << 3 * (H + 2)); r |= t & (p >> (H + 2));
    t = (p >> (H + 2)) & (p >> 2 * (H + 2));
    r |= t & (p << (H + 2)); r |= t & (p >> 3 * (H + 2));
    return r & (BOARD ^ mask);
}

static inline u64 possible(const Pos *P) { return (P->mask + BOTTOM) & BOARD; }
static inline int can_win_next(const Pos *P) { return (winning_cells(P->cur, P->mask) & possible(P)) != 0; }
static inline u64 key_of(const Pos *P) { return P->cur + P->mask + BOTTOM; }
static inline void play(Pos *P, u64 move) { P->cur ^= P->mask; P->mask |= move; P->moves++; }

static u64 non_losing_moves(const Pos *P) {
    u64 poss = possible(P);
    u64 opp_win = winning_cells(P->cur ^ P->mask, P->mask);
    u64 forced = poss & opp_win;
    if (forced) {
        if (forced & (forced - 1)) return 0;   /* two forced blocks: lost */
        poss = forced;
    }
    return poss & ~(opp_win >> 1);
}

/* ---- transposition table ---- */
typedef struct { u64 key; int8_t val; int8_t flag; } Entry;   /* flag 1 = lower bound, 2 = upper bound */
static Entry *table = NULL;
static u64 table_size = 0;
static long long nodes = 0;
static long long budget = 0, node_end = 0;   /* budget: max nodes per solve (0 = unlimited) */
static int aborted = 0;
void c4_set_budget(long long n) { budget = n; }

void c4_init(int log2_entries) {
    if (table) free(table);
    table_size = 1ULL << log2_entries;
    table = (Entry *)calloc(table_size, sizeof(Entry));
}
void c4_reset(void) { if (table) memset(table, 0, table_size * sizeof(Entry)); }
long long c4_nodes(void) { return nodes; }

static int negamax(const Pos *P, int alpha, int beta) {
    nodes++;
    if (node_end && nodes > node_end) { aborted = 1; return 0; }
    if (can_win_next(P)) return (W * H + 1 - P->moves) / 2;
    u64 poss = non_losing_moves(P);
    if (poss == 0) return -(W * H - P->moves) / 2;
    if (P->moves >= W * H - 2) return 0;

    int min = -(W * H - 2 - P->moves) / 2;
    if (alpha < min) { alpha = min; if (alpha >= beta) return alpha; }
    int max = (W * H - 1 - P->moves) / 2;
    if (beta > max) { beta = max; if (alpha >= beta) return beta; }

    u64 key = key_of(P);
    Entry *e = &table[key % table_size];
    if (e->key == key) {
        if (e->flag == 1) { if (alpha < e->val) { alpha = e->val; if (alpha >= beta) return alpha; } }
        else if (e->flag == 2) { if (beta > e->val) { beta = e->val; if (alpha >= beta) return beta; } }
    }

    u64 moves[W]; int score[W]; int n = 0;
    for (int i = 0; i < W; i++) {
        u64 m = poss & COLMASK(ORDER[i]);
        if (!m) continue;
        int s = __builtin_popcountll(winning_cells(P->cur | m, P->mask));
        int j = n++;
        while (j > 0 && score[j - 1] > s) { score[j] = score[j - 1]; moves[j] = moves[j - 1]; j--; }
        score[j] = s; moves[j] = m;
    }
    for (int i = n - 1; i >= 0; i--) {
        Pos Q = *P; play(&Q, moves[i]);
        int v = -negamax(&Q, -beta, -alpha);
        if (aborted) return 0;
        if (v >= beta) {
            e = &table[key % table_size]; e->key = key; e->val = (int8_t)v; e->flag = 1;
            return v;
        }
        if (v > alpha) alpha = v;
    }
    e = &table[key % table_size]; e->key = key; e->val = (int8_t)alpha; e->flag = 2;
    return alpha;
}

static int solve_pos(const Pos *P, int weak) {
    aborted = 0;
    node_end = budget ? nodes + budget : 0;
    if (can_win_next(P)) return (W * H + 1 - P->moves) / 2;
    int min = -(W * H - P->moves) / 2, max = (W * H + 1 - P->moves) / 2;
    if (weak) { min = -1; max = 1; }
    while (min < max) {
        int med = min + (max - min) / 2;
        if (med <= 0 && min / 2 < med) med = min / 2;
        else if (med >= 0 && max / 2 > med) med = max / 2;
        int r = negamax(P, med, med + 1);
        if (aborted) return -999;            /* ran out of budget: unknown */
        if (r <= med) max = r; else min = r;
    }
    return min;
}

/* Parse a string of column digits. Returns 0 on success, -1 if illegal. */
static int parse(const char *seq, Pos *P) {
    P->cur = 0; P->mask = 0; P->moves = 0;
    for (const char *s = seq; *s; s++) {
        if (*s < '0' || *s > '6') return -1;
        int c = *s - '0';
        u64 m = (P->mask + (1ULL << (c * (H + 1)))) & COLMASK(c);
        if (!m) return -1;                                  /* column full */
        if (winning_cells(P->cur, P->mask) & m) return -1;  /* game already won */
        play(P, m);
    }
    return 0;
}

/* Score of the position for the player to move. */
int c4_solve(const char *seq, int weak) {
    Pos P;
    if (parse(seq, &P)) return -1000;
    return solve_pos(&P, weak);
}

/* Per-column scores for the player to move (out[7]); -1000 = column not playable. */
int c4_scores(const char *seq, int *out, int weak) {
    Pos P;
    if (parse(seq, &P)) return -1;
    u64 poss = possible(&P);
    for (int c = 0; c < W; c++) {
        u64 m = poss & COLMASK(c);
        if (!m) { out[c] = -1000; continue; }
        if (winning_cells(P.cur, P.mask) & m) { out[c] = (W * H + 1 - P.moves) / 2; continue; }
        Pos Q = P; play(&Q, m);
        out[c] = -solve_pos(&Q, weak);
    }
    return 0;
}
