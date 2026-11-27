# src/ai_base.py
"""Base interface and shared utilities for Connect-6 AI search strategies."""

from abc import ABC, abstractmethod
import src.constants as c


class AISearchStrategy(ABC):
    """Abstract base class for Connect-6 AI move-search algorithms."""

    DIRECTIONS = ((1, 0), (0, 1), (1, 1), (1, -1))

    @abstractmethod
    def find_best_move(self, game):
        """Return the best list of move coordinates for the current game state."""

    @staticmethod
    def chebyshev(a, b):
        """Return Chebyshev (L-infinity) distance between two board coordinates."""
        (x1, y1), (x2, y2) = a, b
        return max(abs(x1 - x2), abs(y1 - y2))

    @staticmethod
    def opponent_of(player):
        """Return the opposing stone symbol for the given player."""
        return c.PLAYER if player == c.AI else c.AI

    @classmethod
    def check_threat_at_position(cls, game, x, y, player):
        """Return the maximum consecutive line length if `player` places at (x, y)."""
        original = game.board.grid[x][y]
        game.board.grid[x][y] = player
        size = game.board.size
        max_consecutive = 0

        try:
            for dx, dy in cls.DIRECTIONS:
                count = 1
                for sign in (1, -1):
                    i, j = x + sign * dx, y + sign * dy
                    while 0 <= i < size and 0 <= j < size and game.board.grid[i][j] == player:
                        count += 1
                        i += sign * dx
                        j += sign * dy

                max_consecutive = max(max_consecutive, count)
                if max_consecutive >= 6:
                    return max_consecutive
        finally:
            game.board.grid[x][y] = original

        return max_consecutive
