# src/minimax.py
"""Object-oriented Minimax search strategy for Connect-6."""

import time
from itertools import combinations
import src.constants as c
from src.ai_base import AISearchStrategy

INF = 10**9
DEFAULT_RADIUS = 2
HUMAN_FOCUS_RADIUS = 3   # prefer searching near last human move when available

# Normalization/clamping thresholds used when ordering and at leaves.
# These keep heuristic magnitudes from overwhelming distance/ordering.
_ORDER_LIMIT = 500.0
_LEAF_LIMIT = 10000.0


def _chebyshev(a, b):
    return AISearchStrategy.chebyshev(a, b)


def board_state_key(game):
    """Return a deterministic string key representing the board and turn state."""
    rows = [''.join(row) for row in game.board.grid]
    return '|'.join(rows) + f":{game.current_player}:{int(game.first_move)}"


def is_win_after_placement(sim_game, coord):
    """Return True if placing at `coord` wins the game in `sim_game`."""
    x, y = coord
    return sim_game.check_winner(x, y)


def _clamp_for_ordering(v: float) -> float:
    """Clamp probe scores used for ordering to [-_ORDER_LIMIT, _ORDER_LIMIT]."""
    try:
        if v != v:  # NaN guard
            return 0.0
        if v > _ORDER_LIMIT:
            return _ORDER_LIMIT
        if v < -_ORDER_LIMIT:
            return -_ORDER_LIMIT
        return float(v)
    except Exception:
        return 0.0


def _clamp_for_leaf(v: float) -> float:
    """Clamp leaf evaluation values to a bounded range but preserve infinities."""
    try:
        if abs(v) >= INF:
            return v
        if v > _LEAF_LIMIT:
            return _LEAF_LIMIT
        if v < -_LEAF_LIMIT:
            return -_LEAF_LIMIT
        return float(v)
    except Exception:
        return 0.0


class MinimaxAI(AISearchStrategy):
    """Minimax game-tree search strategy with threat detection and beam ordering."""

    def __init__(
        self,
        depth=c.MINI_MAX_DEPTH,
        radius=DEFAULT_RADIUS,
        max_candidates=c.MAX_CANDIDATES,
        max_second_per_first=6,
        verbose=True,
    ):
        self.depth = depth
        self.radius = radius
        self.max_candidates = max_candidates
        self.max_second_per_first = max_second_per_first
        self.verbose = verbose
        self.node_count = 0
        self.last_stats = {}
        self._sim_cache = {}
        self._probe_cache = {}

    def find_best_move(self, game):
        """Return the best move list for `game` using Minimax."""
        _, best_moves = self.search(game, maximizing_player=True)
        return best_moves

    @staticmethod
    def get_candidate_moves(game, radius=DEFAULT_RADIUS):
        """Return empty cells within Chebyshev `radius` of any occupied cell."""
        grid = game.board.grid
        n = game.board.size
        occupied = {(i, j) for i in range(n) for j in range(n) if grid[i][j] != c.EMPTY}
        avail = set(game.get_available_moves())

        if not occupied:
            return sorted(avail)

        candidates = {
            cell
            for cell in avail
            if any(AISearchStrategy.chebyshev(cell, occ) <= radius for occ in occupied)
        }
        return sorted(candidates) if candidates else sorted(avail)

    @staticmethod
    def _human_origins(game):
        """Return the set of recent human move coordinates on `game`."""
        origins = set()
        if hasattr(game, "last_human_moves") and game.last_human_moves:
            origins.update(game.last_human_moves)
        elif getattr(game, "last_row", None) is not None and getattr(game, "last_col", None) is not None:
            origins.add((game.last_row, game.last_col))
        return origins

    @classmethod
    def get_candidate_moves_near_last_human(cls, game, radius=HUMAN_FOCUS_RADIUS):
        """Return empty cells within Chebyshev `radius` of the last human move(s)."""
        origins = cls._human_origins(game)
        if not origins:
            return set()

        avail = set(game.get_available_moves())
        return {
            cell
            for cell in avail
            if any(AISearchStrategy.chebyshev(cell, origin) <= radius for origin in origins)
        }

    def simulate(self, game, moves, player):
        """Return a cached simulated Connect6Game copy after applying `moves`."""
        bk = board_state_key(game)
        key = (bk, tuple(moves), player)
        if key not in self._sim_cache:
            self._sim_cache[key] = game.make_move_copy(moves, player)
        return self._sim_cache[key]

    def probe_score(self, game):
        """Return a cached heuristic evaluation score for `game`."""
        bk = board_state_key(game)
        if bk not in self._probe_cache:
            self._probe_cache[bk] = game.evaluate_position()
        return self._probe_cache[bk]

    def _select_candidates(self, game, avail):
        """Select focused candidates near human moves or general board candidates."""
        avail_set = set(avail)
        focused = self.get_candidate_moves_near_last_human(game, HUMAN_FOCUS_RADIUS)
        if focused:
            candidates = [cell for cell in sorted(focused) if cell in avail_set]
        else:
            candidates = [cell for cell in self.get_candidate_moves(game, self.radius) if cell in avail_set]

        if not candidates:
            candidates = list(avail)
        return candidates

    def _order_root_candidates(self, game, candidates, root_player):
        """Order root candidates by proximity to last human moves and probe score."""
        origins = self._human_origins(game)

        def _dist_to_orig(cell):
            if not origins:
                return float('inf')
            return min(self.chebyshev(cell, o) for o in origins)

        scored = []
        for cell in candidates:
            d = _dist_to_orig(cell)
            try:
                child = self.simulate(game, [cell], root_player)
                ps = self.probe_score(child)
            except Exception:
                ps = 0
            ps_order = _clamp_for_ordering(ps)
            scored.append((d, -ps_order, ps, cell))

        scored.sort()
        ordered = [t[3] for t in scored]
        if self.max_candidates and len(ordered) > self.max_candidates:
            ordered = ordered[:self.max_candidates]
        return ordered

    def _find_single_move_wins(self, game, moves, player):
        """Return the set of single moves in `moves` that immediately win for `player`."""
        wins = set()
        for m in moves:
            try:
                sim = self.simulate(game, [m], player)
                if is_win_after_placement(sim, m):
                    wins.add(m)
            except Exception:
                pass
        return wins

    def _find_two_move_threats(self, game, avail, candidates, opponent):
        """Detect forced two-move opponent wins and return blocking cells."""
        blocking_moves = set()

        try:
            for a, b in combinations(candidates, 2):
                child_ab = self.simulate(game, [a, b], opponent)
                if is_win_after_placement(child_ab, a) or is_win_after_placement(child_ab, b):
                    blocking_moves.add(a)
                    blocking_moves.add(b)
        except Exception:
            pass

        if not blocking_moves:
            try:
                origins = self._human_origins(game)
                for a in avail:
                    child_a = self.simulate(game, [a], opponent)
                    if is_win_after_placement(child_a, a):
                        blocking_moves.add(a)
                        continue

                    local_avail2 = list(child_a.get_available_moves())
                    b_candidates = [
                        b for b in local_avail2
                        if self.chebyshev(b, a) <= HUMAN_FOCUS_RADIUS
                        or any(self.chebyshev(b, o) <= HUMAN_FOCUS_RADIUS for o in origins)
                    ]

                    for b in b_candidates:
                        child_ab = self.simulate(child_a, [b], opponent)
                        if is_win_after_placement(child_ab, b) or is_win_after_placement(child_ab, a):
                            blocking_moves.add(a)
                            blocking_moves.add(b)
                            break
                    if blocking_moves:
                        break
            except Exception:
                pass

        if not blocking_moves and len(avail) <= 120:
            try:
                for a in avail:
                    for b in avail:
                        if b == a:
                            continue
                        child_ab = self.simulate(game, [a, b], opponent)
                        if is_win_after_placement(child_ab, a) or is_win_after_placement(child_ab, b):
                            blocking_moves.add(a)
                            blocking_moves.add(b)
                            break
                    if blocking_moves:
                        break
            except Exception:
                pass

        return blocking_moves

    def _record_stats(self, game, start_time, nodes):
        """Store search statistics on both the instance and the game object."""
        elapsed = time.time() - start_time
        stats = {"time": elapsed, "depth": self.depth, "nodes": nodes}
        self.last_stats = stats
        setattr(game, "_last_minimax_stats", stats)
        return elapsed

    def _recurse(self, node_game, depth_left, maximizing):
        """Recursive Minimax evaluation returning (best_score, best_moves)."""
        self.node_count += 1

        if node_game.is_draw():
            return 0, None
        if depth_left == 0:
            return _clamp_for_leaf(self.probe_score(node_game)), None

        local_avail = list(node_game.get_available_moves())
        local_cand = self._select_candidates(node_game, local_avail)
        if self.max_candidates and len(local_cand) > self.max_candidates:
            local_cand = local_cand[:self.max_candidates]

        if not local_cand:
            return self.probe_score(node_game), None

        best_score = -INF if maximizing else INF
        best_moves = None

        if len(local_cand) >= 2:
            pair_candidates = list(combinations(local_cand, 2))
        else:
            a = local_cand[0]
            pair_candidates = [(a, b) for b in local_avail if b != a]

        scored_pairs = []
        for a, b in pair_candidates:
            child_ab = self.simulate(node_game, [a, b], node_game.current_player)
            if is_win_after_placement(child_ab, b) or is_win_after_placement(child_ab, a):
                return (INF if maximizing else -INF), [a, b]
            ps = self.probe_score(child_ab)
            scored_pairs.append((_clamp_for_ordering(ps), (a, b), child_ab))

        scored_pairs.sort(reverse=maximizing, key=lambda t: t[0])

        max_pairs = self.max_second_per_first * max(1, len(local_cand))
        for i, (_, (a, b), child_ab) in enumerate(scored_pairs):
            if i >= max_pairs:
                break
            sc, _ = self._recurse(child_ab, depth_left - 1, not maximizing)
            if (maximizing and sc > best_score) or (not maximizing and sc < best_score):
                best_score = sc
                best_moves = [a, b]

        return best_score, best_moves

    def search(self, game, maximizing_player=True):
        """Execute Minimax search and return `(best_score, best_moves)`."""
        t0 = time.time()
        required = 1 if game.first_move else 2
        root_player = game.current_player
        opponent = self.opponent_of(root_player)
        avail = list(game.get_available_moves())

        if len(avail) <= required:
            elapsed = self._record_stats(game, t0, 0)
            if self.verbose:
                print(f"[minimax] depth={self.depth} nodes={0} time={elapsed:.4f}s (console)")
            return 0, avail[:required]

        # Check immediate 1-move opponent threats across all available cells
        immediate_blockers = self._find_single_move_wins(game, avail, opponent)
        if immediate_blockers:
            if self.verbose:
                print(f"[minimax] Found immediate threats (blocking required): {sorted(immediate_blockers)}")
            chosen_blockers = sorted(immediate_blockers)[:required]
            self._record_stats(game, t0, 0)
            if self.verbose:
                print(f"[minimax] Returning blockers: {chosen_blockers}")
            return 0, chosen_blockers

        candidates = self._select_candidates(game, avail)
        candidates = self._order_root_candidates(game, candidates, root_player)

        # Check 2-move opponent threats when playing 2-stone turns
        if required == 2:
            blocking_moves = self._find_single_move_wins(game, avail, opponent)
            if not blocking_moves:
                blocking_moves = self._find_two_move_threats(game, avail, candidates, opponent)
            if blocking_moves:
                if self.verbose:
                    print(f"[minimax] Detected opponent immediate/two-move threat. Blocking moves: {sorted(blocking_moves)}")
                chosen_blockers = sorted(blocking_moves)[:required]
                if chosen_blockers:
                    if self.verbose:
                        print(f"[minimax] Returning detected blocking moves as best moves: {chosen_blockers}")
                    self._record_stats(game, t0, 0)
                    return 0, chosen_blockers

        # Check immediate 1-move AI win
        for a in candidates:
            child = self.simulate(game, [a], root_player)
            if is_win_after_placement(child, a):
                self._record_stats(game, t0, 1)
                if self.verbose:
                    print(f"[minimax] Found immediate win: {a}")
                if required == 1:
                    return INF, [a]
                second = next((b for b in candidates if b != a), None)
                if second is None:
                    second = next((b for b in avail if b != a), None)
                return INF, [a, second] if second else [a]

        if required == 1:
            center = (game.board.size // 2, game.board.size // 2)
            first_choice = [center] if center in avail else avail[:1]
            self._record_stats(game, t0, 1)
            return 0, first_choice

        self.node_count = 0
        best_score, best_moves = self._recurse(game, self.depth, maximizing_player)
        elapsed = self._record_stats(game, t0, self.node_count)

        if self.verbose:
            print(f"[minimax] depth={self.depth} nodes={self.node_count} time={elapsed:.4f}s (console)")

        avail_now = set(game.get_available_moves())
        if not best_moves or len(best_moves) != required or any(m not in avail_now for m in best_moves):
            return best_score, list(avail)[:required]

        return best_score, best_moves


# Shared caches and backward-compatible module-level helper functions
_SIM_CACHE = {}
_PROBE_CACHE = {}


def get_candidate_moves(game, radius=DEFAULT_RADIUS):
    return MinimaxAI.get_candidate_moves(game, radius=radius)


def get_candidate_moves_near_last_human(game, radius=HUMAN_FOCUS_RADIUS):
    return MinimaxAI.get_candidate_moves_near_last_human(game, radius=radius)


def simulate(game, moves, player):
    bk = board_state_key(game)
    key = (bk, tuple(moves), player)
    if key not in _SIM_CACHE:
        _SIM_CACHE[key] = game.make_move_copy(moves, player)
    return _SIM_CACHE[key]


def probe_score_cached(game):
    bk = board_state_key(game)
    if bk not in _PROBE_CACHE:
        _PROBE_CACHE[bk] = game.evaluate_position()
    return _PROBE_CACHE[bk]


def minimax(
    game,
    depth,
    maximizing_player,
    radius=DEFAULT_RADIUS,
    max_candidates=c.MAX_CANDIDATES,
    max_second_per_first=6,
    verbose=True,
):
    """Run Minimax search using `MinimaxAI` (backward-compatible entry point)."""
    solver = MinimaxAI(
        depth=depth,
        radius=radius,
        max_candidates=max_candidates,
        max_second_per_first=max_second_per_first,
        verbose=verbose,
    )
    return solver.search(game, maximizing_player=maximizing_player)
