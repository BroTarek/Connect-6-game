# src/heuristics.py
"""
Object-oriented heuristic evaluation strategies for Connect-6 game positions.

Each evaluation approach inherits from `BaseHeuristic` and implements
`evaluate(game_state)`, returning a score from the AI's perspective
(positive = favorable for AI, negative = favorable for human).
"""

from abc import ABC, abstractmethod
import src.constants as c


class BaseHeuristic(ABC):
    """Abstract base class for Connect-6 board evaluation heuristics."""

    DIRECTIONS = ((1, 0), (0, 1), (1, 1), (1, -1))
    SEQUENCE_WEIGHTS = {2: 1, 3: 10, 4: 100, 5: 1000}

    @abstractmethod
    def evaluate(self, game_state):
        """Evaluate `game_state` and return a numerical score from the AI's perspective."""

    @staticmethod
    def _scan_line(grid, size, i, j, dx, dy, player):
        """Count contiguous stones of `player` through (i, j) along (dx, dy).

        Returns:
            tuple: (count, (end1_x, end1_y), (end2_x, end2_y)) where the two
            endpoints are the exact cells immediately beyond each end of the run.
        """
        count = 1

        x, y = i + dx, j + dy
        while 0 <= x < size and 0 <= y < size and grid[x][y] == player:
            count += 1
            x += dx
            y += dy
        end1 = (x, y)

        x, y = i - dx, j - dy
        while 0 <= x < size and 0 <= y < size and grid[x][y] == player:
            count += 1
            x -= dx
            y -= dy
        end2 = (x, y)

        return count, end1, end2

    @staticmethod
    def _is_open_cell(grid, size, coord):
        """Return True if `coord` is in bounds and currently empty."""
        x, y = coord
        return 0 <= x < size and 0 <= y < size and grid[x][y] == c.EMPTY


class SequenceHeuristic(BaseHeuristic):
    """First heuristic: scores sequences of 2-5 consecutive stones."""

    def evaluate(self, game_state):
        score = 0
        grid = game_state.board.grid
        size = game_state.board.size

        for i in range(size):
            for j in range(size):
                player = grid[i][j]
                if player == c.EMPTY:
                    continue

                is_ai = player == game_state.ai_player
                for dx, dy in self.DIRECTIONS:
                    count, _, _ = self._scan_line(grid, size, i, j, dx, dy, player)
                    if 2 <= count <= 5:
                        weight = self.SEQUENCE_WEIGHTS.get(count, 0)
                        score += weight if is_ai else -weight

        return score


class ThreatAndCenterHeuristic(BaseHeuristic):
    """Second heuristic: evaluates sequences, open-4/5 threats, and center control."""

    THREAT_WEIGHT = 7000
    OPEN4_WEIGHT = 1500
    CENTER_BONUS_FACTOR = 0.1

    def evaluate(self, game_state):
        score = 0
        grid = game_state.board.grid
        size = game_state.board.size
        center = size // 2

        for i in range(size):
            for j in range(size):
                player = grid[i][j]
                if player == c.EMPTY:
                    continue

                is_ai = player == game_state.ai_player
                distance_from_center = abs(i - center) + abs(j - center)
                center_bonus = (size - distance_from_center) * self.CENTER_BONUS_FACTOR
                score += center_bonus if is_ai else -center_bonus

                for dx, dy in self.DIRECTIONS:
                    count, end1, end2 = self._scan_line(grid, size, i, j, dx, dy, player)

                    if 2 <= count <= 5:
                        weight = self.SEQUENCE_WEIGHTS.get(count, 0)
                        score += weight if is_ai else -weight

                    if count in (4, 5):
                        if self._is_open_cell(grid, size, end1) or self._is_open_cell(grid, size, end2):
                            bonus = self.THREAT_WEIGHT if count == 5 else self.OPEN4_WEIGHT
                            score += bonus if is_ai else -bonus

        return score


Heuristic1 = SequenceHeuristic
Heuristic2 = ThreatAndCenterHeuristic

HEURISTIC_REGISTRY = {
    c.EVAL1: SequenceHeuristic,
    c.EVAL2: ThreatAndCenterHeuristic,
}


def create_heuristic(name=c.EVAL1):
    """Factory returning a BaseHeuristic instance for the given heuristic name."""
    if isinstance(name, BaseHeuristic):
        return name
    heuristic_cls = HEURISTIC_REGISTRY.get(name, SequenceHeuristic)
    return heuristic_cls()


def heuristic_1(game_state):
    """Evaluate `game_state` using SequenceHeuristic (backward-compatible helper)."""
    return SequenceHeuristic().evaluate(game_state)


def heuristic_2(game_state):
    """Evaluate `game_state` using ThreatAndCenterHeuristic (backward-compatible helper)."""
    return ThreatAndCenterHeuristic().evaluate(game_state)
