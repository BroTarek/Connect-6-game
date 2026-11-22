# tests/test_minimax.py
import unittest
import math
from src.game_logic import Connect6Game
from src.minimax import (
    _chebyshev,
    get_candidate_moves,
    get_candidate_moves_near_last_human,
    board_state_key,
    _clamp_for_ordering,
    _clamp_for_leaf,
    minimax,
)
from src.alpha_beta import AlphaBetaPruning
import src.constants as c


class TestMinimaxHelpers(unittest.TestCase):
    """Unit tests for minimax helper and utility functions."""

    def setUp(self):
        self.game = Connect6Game(size=9, human_player='X', ai_player='O')

    def test_chebyshev_distance(self):
        """Test Chebyshev (L-infinity) distance calculation."""
        self.assertEqual(_chebyshev((0, 0), (0, 0)), 0)
        self.assertEqual(_chebyshev((2, 3), (5, 3)), 3)
        self.assertEqual(_chebyshev((1, 1), (4, 5)), 4)
        self.assertEqual(_chebyshev((10, 2), (2, 10)), 8)

    def test_get_candidate_moves_empty_board(self):
        """On an empty board, candidate moves return all available moves."""
        candidates = get_candidate_moves(self.game, radius=2)
        self.assertEqual(len(candidates), 9 * 9)

    def test_get_candidate_moves_occupied_board(self):
        """Candidates must be within Chebyshev radius of occupied cells."""
        self.game.board.place_move(4, 4, 'X')
        self.game._remove_move(4, 4)
        candidates = get_candidate_moves(self.game, radius=1)

        # For a single stone at (4,4), Chebyshev radius 1 gives at most 8 neighboring cells
        self.assertNotIn((4, 4), candidates)
        self.assertLessEqual(len(candidates), 8)
        for c_pos in candidates:
            self.assertLessEqual(_chebyshev(c_pos, (4, 4)), 1)

    def test_get_candidate_moves_near_last_human(self):
        """Test filtering candidates strictly around last human moves."""
        # No human moves yet
        self.assertEqual(get_candidate_moves_near_last_human(self.game), set())

        # Set last human moves
        self.game.last_human_moves = {(4, 4)}
        near_candidates = get_candidate_moves_near_last_human(self.game, radius=2)
        self.assertGreater(len(near_candidates), 0)
        for c_pos in near_candidates:
            self.assertLessEqual(_chebyshev(c_pos, (4, 4)), 2)

    def test_board_state_key(self):
        """State key must be deterministic and change when board changes."""
        key1 = board_state_key(self.game)
        key2 = board_state_key(self.game)
        self.assertEqual(key1, key2)

        self.game.board.place_move(0, 0, 'X')
        key3 = board_state_key(self.game)
        self.assertNotEqual(key1, key3)

    def test_clamping_functions(self):
        """Test ordering and leaf score clamping."""
        self.assertEqual(_clamp_for_ordering(100.0), 100.0)
        self.assertEqual(_clamp_for_ordering(10000.0), 500.0)
        self.assertEqual(_clamp_for_ordering(-10000.0), -500.0)
        self.assertEqual(_clamp_for_ordering(float('nan')), 0.0)

        self.assertEqual(_clamp_for_leaf(200.0), 200.0)
        self.assertEqual(_clamp_for_leaf(50000.0), 10000.0)
        self.assertEqual(_clamp_for_leaf(-50000.0), -10000.0)
        # Infinite values should be preserved
        self.assertEqual(_clamp_for_leaf(10**9), 10**9)


class TestMinimaxAlgorithm(unittest.TestCase):
    """Unit tests for the Minimax search algorithm."""

    def setUp(self):
        self.game = Connect6Game(size=9, human_player='X', ai_player='O')
        self.game.first_move = False
        self.game.current_player = 'O'

    def test_minimax_returns_two_valid_moves(self):
        """Minimax returns 2 available moves for non-first turn."""
        self.game.board.place_move(4, 4, 'X')
        self.game._remove_move(4, 4)

        score, moves = minimax(self.game, depth=1, maximizing_player=True, verbose=False)
        self.assertIsInstance(moves, list)
        self.assertEqual(len(moves), 2)
        for m in moves:
            self.assertIn(m, self.game.available_moves)

    def test_minimax_immediate_threat_blocking(self):
        """When the opponent has 5-in-a-row, Minimax must detect and return blocking moves."""
        # Opponent 'X' has 5-in-a-row on row 4, cols 2-6: (4,2) to (4,6)
        # Empty ends at (4,1) and (4,7)
        for col in range(2, 7):
            self.game.board.place_move(4, col, 'X')
            self.game._remove_move(4, col)

        score, blockers = minimax(self.game, depth=1, maximizing_player=True, verbose=False)
        self.assertEqual(len(blockers), 2)
        # Must include at least one of the immediate threat ends
        threat_ends = {(4, 1), (4, 7)}
        self.assertTrue(any(b in threat_ends for b in blockers))


class TestAlphaBetaAlgorithm(unittest.TestCase):
    """Unit tests for Alpha-Beta Pruning implementation."""

    def setUp(self):
        self.game = Connect6Game(size=9, human_player='X', ai_player='O')

    def test_alpha_beta_first_move_returns_center(self):
        """On the very first move, Alpha-Beta plays the center of the board."""
        self.game.first_move = True
        ab = AlphaBetaPruning(self.game, heuristic='heuristic_1', max_depth=1)
        moves = ab.find_best_move(self.game)

        center = self.game.board.size // 2
        self.assertEqual(moves, [(center, center)])

    def test_alpha_beta_returns_two_valid_moves(self):
        """Alpha-Beta returns 2 available moves on standard turn."""
        self.game.first_move = False
        self.game.current_player = 'O'
        self.game.board.place_move(4, 4, 'X')
        self.game._remove_move(4, 4)

        ab = AlphaBetaPruning(self.game, heuristic='heuristic_1', max_depth=1)
        moves = ab.find_best_move(self.game)

        self.assertEqual(len(moves), 2)
        for m in moves:
            self.assertIn(m, self.game.available_moves)

    def test_alpha_beta_check_threat_at_position(self):
        """check_threat_at_position measures consecutive stones without mutating board."""
        for y in range(3):
            self.game.board.place_move(2, y, 'O')

        ab = AlphaBetaPruning(self.game, heuristic='heuristic_1', max_depth=1)
        # Placing at (2, 3) creates 4 consecutive stones
        threat = ab.check_threat_at_position(self.game, 2, 3, 'O')
        self.assertEqual(threat, 4)
        # Cell (2, 3) must remain empty after check
        self.assertEqual(self.game.board.grid[2][3], '.')

    def test_alpha_beta_immediate_block(self):
        """Alpha-Beta must block opponent 5-in-a-row threat."""
        self.game.first_move = False
        self.game.current_player = 'O'

        # Human has 5 stones at (5, 2) through (5, 6)
        for y in range(2, 7):
            self.game.board.place_move(5, y, 'X')
            self.game._remove_move(5, y)

        ab = AlphaBetaPruning(self.game, heuristic='heuristic_2', max_depth=1)
        moves = ab.find_best_move(self.game)

        self.assertEqual(len(moves), 2)
        threat_ends = {(5, 1), (5, 7)}
        self.assertTrue(any(m in threat_ends for m in moves))


if __name__ == '__main__':
    unittest.main()
