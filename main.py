# main.py
"""Application entry point for Connect-6 (GUI and Console modes)."""

import sys
import tkinter as tk
import src.constants as c
from src.game_logic import Connect6Game
from src.gui import Connect6GUI


class Connect6Console:
    """Interactive command-line interface for playing Connect-6 against the AI."""

    ALGORITHM_OPTIONS = {
        1: ('minimax', 'heuristic_1'),
        2: ('minimax', 'heuristic_2'),
        3: ('alpha_beta', 'heuristic_1'),
        4: ('alpha_beta', 'heuristic_2'),
    }

    def __init__(self, board_size=None):
        self.board_size = board_size
        self.game = None

    @staticmethod
    def get_board_size():
        """Prompt user for a valid board size (6-19)."""
        print("=== Connect 6 Game Setup ===\n")
        while True:
            try:
                size = int(input("Enter board size (6-19): ").strip())
                if 6 <= size <= 19:
                    return size
                print("Board size must be between 6 and 19. Please try again.")
            except ValueError:
                print("Invalid input. Please enter a number between 6 and 19.")

    @classmethod
    def get_ai_algorithm_choice(cls):
        """Prompt user to pick the AI search algorithm and heuristic for the turn."""
        print("\nChoose AI Algorithm for this turn:")
        print("1 - Minimax with 1st heuristic")
        print("2 - Minimax with 2nd heuristic")
        print("3 - Alpha-Beta with 1st heuristic")
        print("4 - Alpha-Beta with 2nd heuristic\n")

        while True:
            try:
                choice_num = int(input("Enter your choice (1-4): ").strip())
                if choice_num in cls.ALGORITHM_OPTIONS:
                    return cls.ALGORITHM_OPTIONS[choice_num]
                print("Invalid choice. Please enter 1, 2, 3, or 4.")
            except ValueError:
                print("Invalid input. Please enter a number.")

    def _print_result(self):
        """Print final game outcome after a win or draw."""
        if self.game.is_draw():
            print("It's a draw!")
        elif self.game.current_player == self.game.human_player:
            print("Congratulations! You win!")
        else:
            print(f"AI (Player {self.game.ai_player}) wins!")

    def _handle_human_turn(self):
        """Prompt human for moves and play the turn. Returns True if game ended."""
        print(f"\nYour turn (Player {self.game.human_player})")
        required_moves = 1 if self.game.first_move else 2
        size = self.game.board.size
        print(f"Place {required_moves} stone(s). Enter coordinates as 'x y' (1-{size}).")

        moves = []
        for i in range(required_moves):
            try:
                raw = input(f"Move {i + 1}: ").strip()
                x_input, y_input = map(int, raw.split())
            except ValueError:
                print(
                    f"Invalid input. You must enter two integers between 1 and {size}, "
                    "separated by space."
                )
                self.game.board.display()
                return False
            moves.append((x_input - 1, y_input - 1))

        confirmation_moves = ", ".join(f"({x + 1}, {y + 1})" for x, y in moves)
        if input(f"Confirm moves {confirmation_moves}? (y/n): ").strip().lower() not in ("y", "yes"):
            print("Move cancelled. Please re-enter your coordinates.")
            return False

        if self.game.play_turn(moves):
            self._print_result()
            return True

        self.game.board.display()
        return False

    def _handle_ai_turn(self):
        """Configure AI, compute move, and play the turn. Returns True if game ended."""
        algorithm, heuristic = self.get_ai_algorithm_choice()
        self.game.set_ai_config(algorithm, heuristic)
        print(f"\nAI (Player {self.game.ai_player}) is thinking using {algorithm} with {heuristic}...")

        ai_depth = c.MINI_MAX_DEPTH if algorithm == 'minimax' else c.ALPHA_BETA_DEPTH
        ai_moves = self.game.get_ai_move(depth=ai_depth)
        if not ai_moves:
            print("AI Error: No valid moves available!")
            return True

        moves_text = ", ".join(f"({x + 1}, {y + 1})" for x, y in ai_moves)
        print(f"AI plays: {moves_text}")

        if self.game.play_turn(ai_moves):
            self.game.board.display()
            self._print_result()
            return True

        self.game.board.display()
        return False

    def run(self):
        """Start the interactive console game loop."""
        board_size = self.board_size if self.board_size is not None else self.get_board_size()
        self.game = Connect6Game(
            size=board_size,
            human_player='X',
            ai_player='O',
            ai_algorithm='minimax',
            heuristic='heuristic_1',
        )

        print("\n=== Connect 6 - Player vs AI ===")
        print(f"Board size: {board_size}x{board_size}")
        print(f"You are Player {self.game.human_player}, AI is Player {self.game.ai_player}\n")
        self.game.board.display()

        while True:
            if self.game.is_human_turn():
                if self._handle_human_turn():
                    break
            elif self.game.is_ai_turn():
                if self._handle_ai_turn():
                    break


def main_console():
    """Run the game in console mode (Player vs AI)."""
    Connect6Console().run()


def main_gui():
    """Run the game in GUI mode."""
    try:
        board_size = Connect6GUI.get_board_size_dialog()
    except tk.TclError as e:
        print(f"Cannot launch GUI: {e}")
        print("A graphical display server ($DISPLAY) is required for GUI mode.")
        print("To run the game in console/terminal mode, use:")
        print("    python3 main.py --console")
        return

    if board_size is None:
        print("Game cancelled.")
        return
    app = Connect6GUI(size=board_size)
    app.run()


def main():
    """Main entry point. Launches GUI by default, or console if --console flag is used."""
    if len(sys.argv) > 1 and sys.argv[1] == "--console":
        main_console()
    else:
        main_gui()


if __name__ == "__main__":
    main()
