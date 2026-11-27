# tests/test_heuristics.py
import unittest
from src.game_logic import Connect6Game
from src.heuristics import (
    BaseHeuristic,
    SequenceHeuristic,
    ThreatAndCenterHeuristic,
    create_heuristic,
    heuristic_1,
    heuristic_2,
)


class TestHeuristics(unittest.TestCase):
    """Unit tests for Connect-6 heuristic evaluation functions."""

    def setUp(self):
        self.game = Connect6Game(size=19, human_player='X', ai_player='O')

    def test_heuristic_1_empty_board(self):
        """An empty board should evaluate to a neutral score (0)."""
        score = heuristic_1(self.game)
        self.assertEqual(score, 0)

    def test_heuristic_1_ai_vs_human_scoring(self):
        """AI sequences produce positive score; human sequences produce negative score."""
        # AI sequence of 3 stones
        for y in range(3):
            self.game.board.place_move(5, y, 'O')
        ai_score = heuristic_1(self.game)
        self.assertGreater(ai_score, 0)

        # Reset and place identical human sequence of 3 stones
        self.setUp()
        for y in range(3):
            self.game.board.place_move(5, y, 'X')
        human_score = heuristic_1(self.game)
        self.assertLess(human_score, 0)

        # Scores should have equal magnitude and opposite signs
        self.assertEqual(ai_score, -human_score)

    def test_heuristic_1_longer_sequences_have_higher_weight(self):
        """Longer sequences must yield strictly higher weights."""
        # 2 in a row
        self.setUp()
        self.game.board.place_move(0, 0, 'O')
        self.game.board.place_move(0, 1, 'O')
        score_2 = heuristic_1(self.game)

        # 3 in a row
        self.setUp()
        for y in range(3):
            self.game.board.place_move(0, y, 'O')
        score_3 = heuristic_1(self.game)

        # 4 in a row
        self.setUp()
        for y in range(4):
            self.game.board.place_move(0, y, 'O')
        score_4 = heuristic_1(self.game)

        self.assertGreater(score_3, score_2)
        self.assertGreater(score_4, score_3)

    def test_heuristic_1_all_four_directions(self):
        """Heuristic 1 must evaluate horizontal, vertical, diagonal, and anti-diagonal lines."""
        # Horizontal
        self.setUp()
        for y in range(3):
            self.game.board.place_move(2, y, 'O')
        h_score = heuristic_1(self.game)

        # Vertical
        self.setUp()
        for x in range(3):
            self.game.board.place_move(x, 2, 'O')
        v_score = heuristic_1(self.game)

        # Main diagonal
        self.setUp()
        for i in range(3):
            self.game.board.place_move(i, i, 'O')
        d_score = heuristic_1(self.game)

        # Anti-diagonal
        self.setUp()
        for i in range(3):
            self.game.board.place_move(2 - i, i, 'O')
        ad_score = heuristic_1(self.game)

        self.assertGreater(h_score, 0)
        self.assertGreater(v_score, 0)
        self.assertGreater(d_score, 0)
        self.assertGreater(ad_score, 0)

    def test_heuristic_2_center_control_bonus(self):
        """Heuristic 2 must value center stones more than corner stones."""
        center = self.game.board.size // 2

        # Stone at center
        self.setUp()
        self.game.board.place_move(center, center, 'O')
        center_score = heuristic_2(self.game)

        # Stone at corner
        self.setUp()
        self.game.board.place_move(0, 0, 'O')
        corner_score = heuristic_2(self.game)

        self.assertGreater(center_score, corner_score)

    def test_heuristic_2_threat_detection(self):
        """Heuristic 2 detects 5-in-a-row threats with very high weights."""
        # Open 5-in-a-row for AI
        for y in range(2, 7):
            self.game.board.place_move(9, y, 'O')
        ai_threat_score = heuristic_2(self.game)

        # Reset and create open 5-in-a-row for Human
        self.setUp()
        for y in range(2, 7):
            self.game.board.place_move(9, y, 'X')
        human_threat_score = heuristic_2(self.game)

        self.assertGreater(ai_threat_score, 5000)
        self.assertLess(human_threat_score, -5000)

    def test_game_evaluate_position_integration(self):
        """Game's evaluate_position method should properly route to chosen heuristic."""
        for y in range(3):
            self.game.board.place_move(9, y, 'O')

        self.game.set_ai_config('minimax', 'heuristic_1')
        score_1 = self.game.evaluate_position()
        self.assertEqual(score_1, heuristic_1(self.game))

        self.game.set_ai_config('minimax', 'heuristic_2')
        score_2 = self.game.evaluate_position()
        self.assertEqual(score_2, heuristic_2(self.game))

    def test_heuristic_2_closed_five_not_scored_as_open_threat(self):
        """A 5-in-a-row blocked on both ends must not receive the open-threat bonus."""
        # Blocked 5-in-a-row for AI: X at (4, 1) and (4, 7), O at (4, 2)..(4, 6)
        # Cells (4, 0) and (4, 8) remain empty (where the old overshoot bug looked).
        self.game.board.place_move(4, 1, 'X')
        self.game.board.place_move(4, 7, 'X')
        for y in range(2, 7):
            self.game.board.place_move(4, y, 'O')

        score = heuristic_2(self.game)
        # 5 stones * weight 1000 = 5000 (plus small center bonuses); must be well below 5000 + 7000
        self.assertLess(score, 6000)

    def test_oop_heuristic_hierarchy_and_factory(self):
        """Test BaseHeuristic subclasses and create_heuristic factory."""
        h1 = create_heuristic('heuristic_1')
        h2 = create_heuristic('heuristic_2')
        self.assertIsInstance(h1, BaseHeuristic)
        self.assertIsInstance(h1, SequenceHeuristic)
        self.assertIsInstance(h2, BaseHeuristic)
        self.assertIsInstance(h2, ThreatAndCenterHeuristic)

        for y in range(3):
            self.game.board.place_move(4, y, 'O')
        self.assertEqual(h1.evaluate(self.game), heuristic_1(self.game))
        self.assertEqual(h2.evaluate(self.game), heuristic_2(self.game))


if __name__ == '__main__':
    unittest.main()
