import math
import copy
import time
import tracemalloc
from functools import wraps


def analyze_complexity(func):
    """Decorator to analyze time and space complexity of functions"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Start memory tracking
        tracemalloc.start()
        start_memory = tracemalloc.get_traced_memory()[0]

        # Time measurement
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()

        # Memory measurement
        current_memory = tracemalloc.get_traced_memory()[0]
        memory_diff = current_memory - start_memory
        tracemalloc.stop()

        # Calculate complexities
        time_taken = end_time - start_time
        memory_used = memory_diff / 1024  # Convert to KB

        print(f"\n{'=' * 50}")
        print(f"Function: {func.__name__}")
        print(f"Time taken: {time_taken:.6f} seconds")
        print(f"Memory used: {memory_used:.2f} KB")

        # Check if first argument exists and has required attributes
        if args and hasattr(args[0], 'ROWS') and hasattr(args[0], 'COLS'):
            n = args[0].ROWS * args[0].COLS
            if time_taken < 0.001:
                time_complexity = "O(1) - Constant"
            elif time_taken < 0.01:
                time_complexity = "O(log n) - Logarithmic"
            elif time_taken < 0.1:
                time_complexity = "O(n) - Linear"
            elif time_taken < 1:
                time_complexity = "O(n log n)"
            else:
                time_complexity = "O(n²) or higher"

            print(f"Estimated time complexity: {time_complexity}")
            print(f"Board size (n): {n}")
        print(f"{'=' * 50}\n")

        return result

    return wrapper



class DropTokenGame:
    def __init__(self):
        self.ROWS = 5
        self.COLS = 5
        self.EMPTY = 0
        self.HUMAN = 1
        self.AI = 2
        self.board = [[self.EMPTY for _ in range(self.COLS)] for _ in range(self.ROWS)]
        self.current_player = self.HUMAN
        
    def display_board(self):
        """Display the current board state"""
        print("\nCurrent Board:")
        print("  ", end="")
        for col in range(self.COLS):
            print(f"{col+1} ", end="")
        print()
        
        for row in range(self.ROWS):
            print(f"{row+1} ", end="")
            for col in range(self.COLS):
                if self.board[row][col] == self.EMPTY:
                    print(". ", end="")
                elif self.board[row][col] == self.HUMAN:
                    print("H ", end="")
                else:
                    print("A ", end="")
            print()
        print()
    
    def is_valid_move(self, col):
        """Check if a move is valid (column not full)"""
        return 0 <= col < self.COLS and self.board[0][col] == self.EMPTY
    
    def make_move(self, col, player):
        """Drop a token in the specified column"""
        for row in range(self.ROWS - 1, -1, -1):
            if self.board[row][col] == self.EMPTY:
                self.board[row][col] = player
                return row
        return -1
    
    def undo_move(self, col):
        """Remove the top token from a column"""
        for row in range(self.ROWS):
            if self.board[row][col] != self.EMPTY:
                self.board[row][col] = self.EMPTY
                return
    
    def check_winner(self):
        """Check if there's a winner (3 in a row)"""
        # Check horizontal
        for row in range(self.ROWS):
            for col in range(self.COLS - 2):
                if (self.board[row][col] != self.EMPTY and
                    self.board[row][col] == self.board[row][col+1] == self.board[row][col+2]):
                    return self.board[row][col]
        
        # Check vertical
        for row in range(self.ROWS - 2):
            for col in range(self.COLS):
                if (self.board[row][col] != self.EMPTY and
                    self.board[row][col] == self.board[row+1][col] == self.board[row+2][col]):
                    return self.board[row][col]
        
        # Check diagonal (top-left to bottom-right)
        for row in range(self.ROWS - 2):
            for col in range(self.COLS - 2):
                if (self.board[row][col] != self.EMPTY and
                    self.board[row][col] == self.board[row+1][col+1] == self.board[row+2][col+2]):
                    return self.board[row][col]
        
        # Check diagonal (top-right to bottom-left)
        for row in range(self.ROWS - 2):
            for col in range(2, self.COLS):
                if (self.board[row][col] != self.EMPTY and
                    self.board[row][col] == self.board[row+1][col-1] == self.board[row+2][col-2]):
                    return self.board[row][col]
        
        return self.EMPTY
    
    def is_board_full(self):
        """Check if the board is full (draw condition)"""
        return all(self.board[0][col] != self.EMPTY for col in range(self.COLS))
    
    def get_valid_moves(self):
        """Get all valid column moves"""
        return [col for col in range(self.COLS) if self.is_valid_move(col)]
    
    def evaluate_position(self):
        """
        Static evaluation function for the current board position.
        
        Strategy:
        1. Win/Loss: +1000/-1000 for immediate wins
        2. Two in a row with space: +10/-10
        3. Center control: +5/-5 for center column
        4. Blocking opponent: +20/-20
        
        This evaluation prioritizes winning, then blocking opponent wins,
        then creating opportunities, and finally controlling the center.
        """
        winner = self.check_winner()
        if winner == self.AI:
            return 1000
        elif winner == self.HUMAN:
            return -1000
        
        score = 0
        
        # Evaluate all possible 3-in-a-row positions
        score += self.evaluate_lines(self.AI) - self.evaluate_lines(self.HUMAN)
        
        # Center column preference
        center_col = self.COLS // 2
        for row in range(self.ROWS):
            if self.board[row][center_col] == self.AI:
                score += 3
            elif self.board[row][center_col] == self.HUMAN:
                score -= 3
        
        return score
    
    def evaluate_lines(self, player):
        """Evaluate all possible 3-token lines for a player"""
        score = 0
        
        # Check all possible 3-token windows
        # Horizontal
        for row in range(self.ROWS):
            for col in range(self.COLS - 2):
                window = [self.board[row][col+i] for i in range(3)]
                score += self.score_window(window, player)
        
        # Vertical
        for row in range(self.ROWS - 2):
            for col in range(self.COLS):
                window = [self.board[row+i][col] for i in range(3)]
                score += self.score_window(window, player)
        
        # Diagonal (positive slope)
        for row in range(self.ROWS - 2):
            for col in range(self.COLS - 2):
                window = [self.board[row+i][col+i] for i in range(3)]
                score += self.score_window(window, player)
        
        # Diagonal (negative slope)
        for row in range(self.ROWS - 2):
            for col in range(2, self.COLS):
                window = [self.board[row+i][col-i] for i in range(3)]
                score += self.score_window(window, player)
        
        return score
    
    def score_window(self, window, player):
        """Score a 3-token window"""
        score = 0
        opponent = self.HUMAN if player == self.AI else self.AI
        
        player_count = window.count(player)
        empty_count = window.count(self.EMPTY)
        opponent_count = window.count(opponent)
        
        if player_count == 3:
            score += 100
        elif player_count == 2 and empty_count == 1:
            score += 10
        elif player_count == 1 and empty_count == 2:
            score += 2
        
        # Block opponent
        if opponent_count == 2 and empty_count == 1:
            score -= 80
        
        return score

    @analyze_complexity
    def minimax(self, depth, maximizing_player, alpha=-math.inf, beta=math.inf):
        """
        Minimax algorithm
        Returns (score, best_column)
        """
        winner = self.check_winner()
        if winner == self.AI:
            return 1000 + depth, None
        elif winner == self.HUMAN:
            return -1000 - depth, None
        elif self.is_board_full() or depth == 0:
            return self.evaluate_position(), None
        
        valid_moves = self.get_valid_moves()
        best_col = valid_moves[0] if valid_moves else None
        
        if maximizing_player:
            max_eval = -math.inf
            for col in valid_moves:
                self.make_move(col, self.AI)
                eval_score, _ = self.minimax(depth - 1, False, alpha, beta)
                self.undo_move(col)
                
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_col = col
                
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            
            return max_eval, best_col
        else:
            min_eval = math.inf
            for col in valid_moves:
                self.make_move(col, self.HUMAN)
                eval_score, _ = self.minimax(depth - 1, True, alpha, beta)
                self.undo_move(col)
                
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_col = col
                
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            
            return min_eval, best_col
    
    def get_ai_move(self):
        """Get the best move for AI using minimax with depth 3"""
        _, best_col = self.minimax(3, True)
        return best_col

    def play_game(self):
        """Main game loop"""
        print("Welcome to Drop Token Game!")
        print("You are 'H' and AI is 'A'")
        print("Enter column number (1-5) to drop your token")
        print("Goal: Get 3 tokens in a row (horizontal, vertical, or diagonal)")
        
        self.display_board()
        
        while True:
            if self.current_player == self.HUMAN:
                # Human move
                try:
                    col_input = input(f"Your turn! Enter column (1-{self.COLS}): ")
                    col = int(col_input) - 1
                    
                    if not self.is_valid_move(col):
                        print("Invalid move! Column is full or out of range.")
                        continue
                    
                    row = self.make_move(col, self.HUMAN)
                    print(f"You dropped token in column {col+1}, row {row+1}")
                    
                except (ValueError, IndexError):
                    print("Please enter a valid number!")
                    continue
            
            else:
                # AI move
                print("AI is thinking...")
                col = self.get_ai_move()
                if col is not None:
                    row = self.make_move(col, self.AI)
                    print(f"AI dropped token in column {col+1}, row {row+1}")
                else:
                    print("AI couldn't find a valid move!")
                    break
            
            self.display_board()
            
            # Check game end conditions
            winner = self.check_winner()
            if winner == self.HUMAN:
                print("🎉 Congratulations! You won!")
                break
            elif winner == self.AI:
                print("🤖 AI wins! Better luck next time!")
                break
            elif self.is_board_full():
                print("🤝 It's a draw!")
                break
            
            # Switch players
            self.current_player = self.AI if self.current_player == self.HUMAN else self.HUMAN

def demonstrate_evaluation_function():
    """Demonstrate the evaluation function with a sample game state"""
    print("\n" + "="*50)
    print("EVALUATION FUNCTION DEMONSTRATION")
    print("="*50)
    
    game = DropTokenGame()
    
    # Create a sample game state
    # H H . . .
    # A A A . .  <- AI has 3 in a row (winning position)
    # . . . . .
    # . . . . .
    # . . . . .
    
    game.board[0][0] = game.HUMAN
    game.board[0][1] = game.HUMAN
    game.board[1][0] = game.AI
    game.board[1][1] = game.AI
    game.board[1][2] = game.AI
    
    print("Sample Game State:")
    game.display_board()
    
    score = game.evaluate_position()
    print(f"Evaluation Score: {score}")
    print("\nExplanation:")
    print("- AI has 3 in a row horizontally (row 2): +1000 points")
    print("- This represents a winning position for AI")
    print("- The high positive score indicates AI should prioritize this move")

if __name__ == "__main__":
    # Demonstrate evaluation function
    demonstrate_evaluation_function()
    
    # Play the game
    game = DropTokenGame()
    game.play_game()