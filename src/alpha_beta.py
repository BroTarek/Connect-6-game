# src/alpha_beta.py
"""Object-oriented Alpha-Beta Pruning search strategy for Connect-6."""

import math
import time
import src.constants as c
from src.ai_base import AISearchStrategy
from src.heuristics import create_heuristic


class AlphaBetaPruning(AISearchStrategy):
    """Alpha-Beta Pruning search strategy inheriting from `AISearchStrategy`."""

    def __init__(self, game, heuristic=c.EVAL1, max_depth=c.ALPHA_BETA_DEPTH, limit=20):
        self.game = game
        self.max_depth = max_depth
        self.heuristic = heuristic
        self.heuristic_evaluator = create_heuristic(heuristic)
        self.limit = limit
        self.nodes_explored = 0
        self.nodes_pruned = 0
        self.start_time = None

    def _evaluate_board(self, game, win, draw):
        """Evaluate terminal or leaf game states via the configured heuristic."""
        if win:
            winner = game.board.grid[game.last_row][game.last_col]
            return math.inf if winner == c.AI else -math.inf
        if draw:
            return 0
        return self.heuristic_evaluator.evaluate(game)

    def alpha_beta(self, game, depth, alpha, beta, maximizing_player):
        """Recursive Alpha-Beta search returning the minimax value from `game`."""
        self.nodes_explored += 1

        if game.last_row is not None and game.last_col is not None:
            win = game.check_winner(game.last_row, game.last_col)
        else:
            win = False
        draw = game.is_draw()

        if (not depth) or draw or win:
            return self._evaluate_board(game, win, draw)

        return self._search_node(game, depth, alpha, beta, maximizing_player)

    def _search_node(self, game, depth, alpha, beta, maximizing):
        """Explore 2-stone turn pairs for either MAX (`maximizing=True`) or MIN."""
        best_score = -math.inf if maximizing else math.inf
        player = c.AI if maximizing else c.PLAYER
        win_score = math.inf if maximizing else -math.inf
        orig_r, orig_c = game.last_row, game.last_col

        moves = self.get_prioritized_moves(game, self.limit, player=player)

        for i in range(len(moves)):
            x1, y1 = moves[i]
            self.set_stone(x1, y1, player)
            if game.check_winner(x1, y1):
                self.nodes_explored += 1
                self.remove_stone(x1, y1)
                game.last_row, game.last_col = orig_r, orig_c
                return win_score
            if game.is_draw():
                self.nodes_explored += 1
                self.remove_stone(x1, y1)
                game.last_row, game.last_col = orig_r, orig_c
                return max(best_score, 0) if maximizing else min(best_score, 0)

            for j in range(i + 1, len(moves)):
                x2, y2 = moves[j]
                self.set_stone(x2, y2, player)

                score = self.alpha_beta(game, depth - 1, alpha, beta, not maximizing)

                self.remove_stone(x2, y2)
                game.last_row, game.last_col = x1, y1

                if maximizing:
                    best_score = max(best_score, score)
                    alpha = max(alpha, score)
                else:
                    best_score = min(best_score, score)
                    beta = min(beta, score)

                if beta <= alpha:
                    self.nodes_pruned += 1
                    self.remove_stone(x1, y1)
                    game.last_row, game.last_col = orig_r, orig_c
                    return best_score

            self.remove_stone(x1, y1)

        game.last_row, game.last_col = orig_r, orig_c
        return best_score

    def _max_node(self, game, depth, alpha, beta):
        return self._search_node(game, depth, alpha, beta, maximizing=True)

    def _min_node(self, game, depth, alpha, beta):
        return self._search_node(game, depth, alpha, beta, maximizing=False)

    def get_prioritized_moves(self, game, limit=None, player=None):
        """Prioritize candidates: Win > Block 5 > Block 4 > Attack 5 > Attack 4 > Other."""
        all_moves = game.get_available_moves()
        if not all_moves:
            return []

        if len(all_moves) >= (game.board.size * game.board.size - 5):
            center = game.board.size // 2
            return [
                (x, y) for x, y in all_moves
                if abs(x - center) <= 4 and abs(y - center) <= 4
            ][:limit]

        if player is not None:
            current_player = player
        else:
            current_player = c.AI if game.current_player == c.AI else c.PLAYER
        opponent = self.opponent_of(current_player)

        winning_moves = []
        critical_blocks = []   # Block 5-in-a-row
        important_blocks = []  # Block 4-in-a-row
        strong_attacks = []    # Create 5-in-a-row
        good_attacks = []      # Create 4-in-a-row
        decent_moves = []

        relevant = set()
        size = game.board.size
        grid = game.board.grid
        for x in range(size):
            for y in range(size):
                if grid[x][y] != c.EMPTY:
                    for dx in range(-3, 4):
                        for dy in range(-3, 4):
                            nx, ny = x + dx, y + dy
                            if 0 <= nx < size and 0 <= ny < size and grid[nx][ny] == c.EMPTY:
                                relevant.add((nx, ny))

        moves_to_check = list(relevant) if relevant else all_moves

        for move in moves_to_check:
            x, y = move
            my_threat = self.check_threat_at_position(game, x, y, current_player)
            opp_threat = self.check_threat_at_position(game, x, y, opponent)

            if my_threat >= 6:
                winning_moves.append(move)
            elif opp_threat >= 5:
                critical_blocks.append((move, opp_threat))
            elif opp_threat >= 4:
                important_blocks.append((move, opp_threat))
            elif my_threat >= 5:
                strong_attacks.append((move, my_threat))
            elif my_threat >= 4:
                good_attacks.append((move, my_threat))
            else:
                score = my_threat * 10 + opp_threat * 5
                decent_moves.append((move, score))

        for bucket in (critical_blocks, important_blocks, strong_attacks, good_attacks, decent_moves):
            bucket.sort(key=lambda item: item[1], reverse=True)

        result = (
            winning_moves
            + [m for m, _ in critical_blocks]
            + [m for m, _ in important_blocks]
            + [m for m, _ in strong_attacks]
            + [m for m, _ in good_attacks]
            + [m for m, _ in decent_moves]
        )

        unique = list(dict.fromkeys(result))
        return unique[:limit] if limit else unique

    def find_best_move(self, game):
        """Return the best 1 or 2 moves for `game` using Alpha-Beta Pruning."""
        self.game = game
        self.nodes_explored = 0
        self.nodes_pruned = 0
        self.start_time = time.time()

        if game.first_move:
            center = game.board.size // 2
            return [(center, center)]

        moves = self.get_prioritized_moves(game, self.limit, player=c.AI)
        current_player = c.AI
        opponent = c.PLAYER

        # Immediate win check (1 or 2 stones)
        for i in range(len(moves)):
            x1, y1 = moves[i]
            if self.check_threat_at_position(game, x1, y1, current_player) >= 6:
                return [(x1, y1), moves[0] if moves[0] != (x1, y1) else moves[1]]

            self.set_stone(x1, y1, current_player)
            for j in range(i + 1, len(moves)):
                x2, y2 = moves[j]
                if self.check_threat_at_position(game, x2, y2, current_player) >= 6:
                    self.remove_stone(x1, y1)
                    return [(x1, y1), (x2, y2)]
            self.remove_stone(x1, y1)

        # Critical block check for opponent 5-in-a-row threats
        critical_threats = []
        for move in moves[:15]:
            x, y = move
            threat_level = self.check_threat_at_position(game, x, y, opponent)
            if threat_level >= 5:
                critical_threats.append((move, threat_level))

        if critical_threats:
            critical_threats.sort(key=lambda item: item[1], reverse=True)
            block1 = critical_threats[0][0]
            if len(critical_threats) > 1:
                block2 = critical_threats[1][0]
            else:
                block2 = moves[1] if moves[1] != block1 else moves[2]
            return [block1, block2]

        # Normal Alpha-Beta search
        best_score = -math.inf
        best_moves = None
        alpha = -math.inf
        beta = math.inf
        search_limit = min(len(moves), 20)

        for i in range(search_limit):
            x1, y1 = moves[i]
            self.set_stone(x1, y1, c.AI)

            for j in range(i + 1, search_limit):
                x2, y2 = moves[j]
                self.set_stone(x2, y2, c.AI)

                score = self.alpha_beta(game, self.max_depth - 1, alpha, beta, False)

                self.remove_stone(x2, y2)
                game.last_row, game.last_col = x1, y1

                if score > best_score:
                    best_score = score
                    best_moves = [(x1, y1), (x2, y2)]

                alpha = max(alpha, score)

            self.remove_stone(x1, y1)

        elapsed_time = time.time() - self.start_time
        print(f"==== Alpha Beta at depth {self.max_depth} ====")
        print(f"Nodes Explored: {self.nodes_explored:,}")
        print(f"Pruned {self.nodes_pruned:} times")
        print(f"Time Taken: {elapsed_time:.2f} seconds")
        print(f"Best Move Returned From AlphaBetaPurning Class: {best_moves}")
        print(f"Best Score: {best_score}")
        return best_moves if best_moves else [moves[0], moves[1]]

    def set_stone(self, x, y, stone):
        """Temporarily place a stone during search while keeping board state consistent."""
        self.game.board.place_move(x, y, stone)
        self.game.last_row = x
        self.game.last_col = y
        self.game._remove_move(x, y)

    def remove_stone(self, x, y):
        """Undo a temporary stone placement during search."""
        self.game.board.undo_move(x, y)
        self.game._add_move(x, y)
