# tests/test_game_logic.py
import unittest
from src.board import Board
from src.game_logic import Connect6Game
import src.constants as c


class TestBoard(unittest.TestCase):
    """Unit tests for the Board class."""

    def setUp(self):
        self.board = Board(size=19)

    def test_initialization(self):
        """Test board initialization with default and custom sizes."""
        self.assertEqual(self.board.size, 19)
        self.assertEqual(self.board.empty_cells, 19 * 19)
        self.assertEqual(self.board.grid[0][0], '.')

        custom_board = Board(size=6)
        self.assertEqual(custom_board.size, 6)
        self.assertEqual(custom_board.empty_cells, 36)

    def test_is_valid_move(self):
        """Test validity checks for moves inside and outside bounds."""
        self.assertTrue(self.board.is_valid_move(0, 0))
        self.assertTrue(self.board.is_valid_move(18, 18))

        # Out of bounds
        self.assertFalse(self.board.is_valid_move(-1, 0))
        self.assertFalse(self.board.is_valid_move(0, -1))
        self.assertFalse(self.board.is_valid_move(19, 0))
        self.assertFalse(self.board.is_valid_move(0, 19))

        # Already occupied
        self.board.place_move(5, 5, 'X')
        self.assertFalse(self.board.is_valid_move(5, 5))

    def test_place_move(self):
        """Test stone placement and empty cells count updates."""
        initial_empty = self.board.empty_cells
        success = self.board.place_move(3, 4, 'X')
        self.assertTrue(success)
        self.assertEqual(self.board.grid[3][4], 'X')
        self.assertEqual(self.board.empty_cells, initial_empty - 1)

        # Placing on occupied cell should fail
        fail = self.board.place_move(3, 4, 'O')
        self.assertFalse(fail)
        self.assertEqual(self.board.grid[3][4], 'X')
        self.assertEqual(self.board.empty_cells, initial_empty - 1)

    def test_undo_move(self):
        """Test undoing moves and restoring empty cell count."""
        self.board.place_move(2, 2, 'O')
        empty_after_place = self.board.empty_cells

        success = self.board.undo_move(2, 2)
        self.assertTrue(success)
        self.assertEqual(self.board.grid[2][2], '.')
        self.assertEqual(self.board.empty_cells, empty_after_place + 1)

        # Undoing already empty cell should return False
        self.assertFalse(self.board.undo_move(2, 2))

    def test_is_full_and_get_empty_cells_count(self):
        """Test full board detection and empty cell counting."""
        small_board = Board(size=2)
        self.assertEqual(small_board.get_empty_cells_count(), 4)
        self.assertFalse(small_board.is_full())

        small_board.place_move(0, 0, 'X')
        small_board.place_move(0, 1, 'O')
        small_board.place_move(1, 0, 'X')
        self.assertFalse(small_board.is_full())
        self.assertEqual(small_board.get_empty_cells_count(), 1)

        small_board.place_move(1, 1, 'O')
        self.assertTrue(small_board.is_full())
        self.assertEqual(small_board.get_empty_cells_count(), 0)

    def test_board_copy_isolation(self):
        """Test deep copy isolation between board instances."""
        self.board.place_move(1, 1, 'X')
        board_clone = self.board.copy()

        self.assertEqual(board_clone.grid[1][1], 'X')
        self.assertEqual(board_clone.empty_cells, self.board.empty_cells)

        # Modify clone, original should remain unaffected
        board_clone.place_move(2, 2, 'O')
        self.assertEqual(board_clone.grid[2][2], 'O')
        self.assertEqual(self.board.grid[2][2], '.')
        self.assertNotEqual(board_clone.empty_cells, self.board.empty_cells)


class TestConnect6GameLogic(unittest.TestCase):
    """Unit tests for Connect6Game rules, turns, moves, wins, and draws."""

    def setUp(self):
        self.game = Connect6Game(size=19, human_player='X', ai_player='O')

    def test_initial_state(self):
        """Test initial game state attributes."""
        self.assertEqual(self.game.board.size, 19)
        self.assertEqual(self.game.current_player, 'X')
        self.assertTrue(self.game.first_move)
        self.assertTrue(self.game.is_human_turn())
        self.assertFalse(self.game.is_ai_turn())
        self.assertEqual(len(self.game.get_available_moves()), 19 * 19)

    def test_switch_player(self):
        """Test player turn toggle."""
        self.game.switch_player()
        self.assertEqual(self.game.current_player, 'O')
        self.assertTrue(self.game.is_ai_turn())
        self.assertFalse(self.game.is_human_turn())

        self.game.switch_player()
        self.assertEqual(self.game.current_player, 'X')

    def test_set_ai_config(self):
        """Test updating AI algorithm and heuristic configurations."""
        self.game.set_ai_config('alpha_beta', 'heuristic_2')
        self.assertEqual(self.game.ai_algorithm, 'alpha_beta')
        self.assertEqual(self.game.heuristic, 'heuristic_2')

    def test_validate_moves_atomic_first_turn(self):
        """First turn requires exactly 1 move."""
        # 1 valid move -> should pass
        valid, msg = self.game._validate_moves_atomic([(9, 9)])
        self.assertTrue(valid)
        self.assertEqual(msg, "")

        # 2 moves on first turn -> should fail
        valid, msg = self.game._validate_moves_atomic([(9, 9), (9, 10)])
        self.assertFalse(valid)
        self.assertIn("exactly 1 move", msg)

        # 0 moves -> should fail
        valid, msg = self.game._validate_moves_atomic([])
        self.assertFalse(valid)

    def test_validate_moves_atomic_subsequent_turns(self):
        """Subsequent turns require exactly 2 moves."""
        self.game.first_move = False

        # 2 valid moves -> should pass
        valid, msg = self.game._validate_moves_atomic([(5, 5), (6, 6)])
        self.assertTrue(valid)

        # 1 move on subsequent turn -> should fail
        valid, msg = self.game._validate_moves_atomic([(5, 5)])
        self.assertFalse(valid)
        self.assertIn("exactly 2 move", msg)

        # 3 moves -> should fail
        valid, msg = self.game._validate_moves_atomic([(1, 1), (1, 2), (1, 3)])
        self.assertFalse(valid)

    def test_validate_moves_atomic_bounds_and_occupied(self):
        """Test validation against out-of-bounds, occupied, and duplicates."""
        self.game.first_move = False

        # Out of bounds
        valid, msg = self.game._validate_moves_atomic([(0, 0), (19, 0)])
        self.assertFalse(valid)
        self.assertIn("out of bounds", msg)

        # Already occupied
        self.game.board.place_move(2, 2, 'X')
        valid, msg = self.game._validate_moves_atomic([(2, 2), (3, 3)])
        self.assertFalse(valid)
        self.assertIn("already occupied", msg)

        # Duplicate coordinate in the same turn
        valid, msg = self.game._validate_moves_atomic([(4, 4), (4, 4)])
        self.assertFalse(valid)
        self.assertIn("Duplicate move", msg)

        # Non-integer coordinates
        valid, msg = self.game._validate_moves_atomic([("a", 1), (2, 2)])
        self.assertFalse(valid)
        self.assertIn("integers", msg)

    def test_play_turn_success_and_progression(self):
        """Test playing turns progresses the game and switches players."""
        # Turn 1: Human ('X') places 1 stone
        ended = self.game.play_turn([(9, 9)])
        self.assertFalse(ended)
        self.assertFalse(self.game.first_move)
        self.assertEqual(self.game.current_player, 'O')
        self.assertEqual(self.game.board.grid[9][9], 'X')
        self.assertNotIn((9, 9), self.game.available_moves)

        # Turn 2: AI ('O') places 2 stones
        ended = self.game.play_turn([(9, 10), (9, 11)])
        self.assertFalse(ended)
        self.assertEqual(self.game.current_player, 'X')
        self.assertEqual(self.game.board.grid[9][10], 'O')
        self.assertEqual(self.game.board.grid[9][11], 'O')
        self.assertNotIn((9, 10), self.game.available_moves)
        self.assertNotIn((9, 11), self.game.available_moves)

    def test_play_turn_atomic_rollback(self):
        """Invalid turn should not place any stones or switch players."""
        self.game.first_move = False
        self.game.board.place_move(1, 1, 'X')
        initial_empty = self.game.board.empty_cells

        # One valid move and one invalid (already occupied)
        ended = self.game.play_turn([(1, 2), (1, 1)])
        self.assertFalse(ended)
        self.assertEqual(self.game.current_player, 'X')
        self.assertEqual(self.game.board.grid[1][2], '.')
        self.assertEqual(self.game.board.empty_cells, initial_empty)

    def test_check_winner_horizontal(self):
        """Test winning with 6 stones horizontally."""
        for y in range(2, 8):
            self.game.board.place_move(5, y, 'X')
        self.assertTrue(self.game.check_winner(5, 7))
        self.assertTrue(self.game.check_winner(5, 2))
        self.assertFalse(self.game.check_winner(5, 8))

    def test_check_winner_vertical(self):
        """Test winning with 6 stones vertically."""
        for x in range(3, 9):
            self.game.board.place_move(x, 4, 'O')
        self.assertTrue(self.game.check_winner(3, 4))
        self.assertTrue(self.game.check_winner(8, 4))
        self.assertFalse(self.game.check_winner(2, 4))

    def test_check_winner_main_diagonal(self):
        """Test winning with 6 stones along main diagonal (\\)."""
        for i in range(6):
            self.game.board.place_move(2 + i, 2 + i, 'X')
        self.assertTrue(self.game.check_winner(7, 7))
        self.assertTrue(self.game.check_winner(2, 2))

    def test_check_winner_anti_diagonal(self):
        """Test winning with 6 stones along anti-diagonal (/)."""
        for i in range(6):
            self.game.board.place_move(8 - i, 2 + i, 'O')
        self.assertTrue(self.game.check_winner(8, 2))
        self.assertTrue(self.game.check_winner(3, 7))

    def test_check_winner_five_stones_not_win(self):
        """Five stones in a row must not trigger a win."""
        for y in range(1, 6):
            self.game.board.place_move(10, y, 'X')
        self.assertFalse(self.game.check_winner(10, 5))

    def test_check_winner_invalid_or_empty(self):
        """Checking winner on out-of-bounds or empty cells returns False."""
        self.assertFalse(self.game.check_winner(-1, 0))
        self.assertFalse(self.game.check_winner(0, 0))

    def test_is_draw_full_board(self):
        """Draw when all cells are filled."""
        small_game = Connect6Game(size=2)
        small_game.first_move = False
        for i in range(2):
            for j in range(2):
                small_game.board.place_move(i, j, 'X')
        self.assertTrue(small_game.is_draw())

    def test_is_draw_even_board_one_cell_left(self):
        """Special draw rule: even board with 1 empty cell remaining and 2 moves needed."""
        small_game = Connect6Game(size=4)
        small_game.first_move = False  # requires 2 moves

        # Fill 15 of 16 cells
        for i in range(4):
            for j in range(4):
                if not (i == 3 and j == 3):
                    small_game.board.place_move(i, j, 'X')

        self.assertEqual(small_game.board.get_empty_cells_count(), 1)
        self.assertTrue(small_game.is_draw())

    def test_make_move_copy_isolation(self):
        """Test make_move_copy produces an isolated simulated state."""
        self.game.first_move = False
        simulated = self.game.make_move_copy([(3, 3), (3, 4)], 'O')

        self.assertEqual(simulated.board.grid[3][3], 'O')
        self.assertEqual(simulated.board.grid[3][4], 'O')
        self.assertNotIn((3, 3), simulated.available_moves)

        # Original game must remain unmodified
        self.assertEqual(self.game.board.grid[3][3], '.')
        self.assertEqual(self.game.board.grid[3][4], '.')
        self.assertIn((3, 3), self.game.available_moves)


if __name__ == '__main__':
    unittest.main()
