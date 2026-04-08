# Import any required libraries here
EMPTY = ' '
PLAYER_X = 'X'
PLAYER_O = 'O'

class TicTacToe:
    def __init__(self):
        self.board = [[EMPTY for _ in range(3)] for _ in range(3)]
        self.user = None
        self.computer = None

    def print_board(self):
        print("  0   1   2")
        for index, row in enumerate(self.board):
            print(index, " | ".join(row))
            if index < 2:
                print("  -----------")

            
    def check_winner(self, board, player):
        """Checks if the specified player has won the game."""
        # Check rows and columns
        for i in range(3):
            if board[i][0] == board[i][1] == board[i][2] == player:
                return True
            if board[0][i] == board[1][i] == board[2][i] == player:
                return True
        
        # Check diagonals
        if board[0][0] == board[1][1] == board[2][2] == player:
            return True
        if board[0][2] == board[1][1] == board[2][0] == player:
            return True
            
        return False
        
    def evaluate(self, board, depth):
        if self.check_winner(board, self.computer):
            return 10 + depth
        elif self.check_winner(board, self.user):
            return -10 - depth
        else:
            return 0
        
    def is_board_full(self, board):
        """Checks if there are no empty spaces left on the board."""
        return all(cell != EMPTY for row in board for cell in row)
    
    def get_available_moves(self, board):
        """Returns a list of tuples (row, col) for all empty cells."""
        return [(r, c) for r in range(3) for c in range(3) if board[r][c] == EMPTY]

    def minimax(self, board, depth, maximizing_player, alpha, beta):
        """Minimax with alpha-beta pruning algorithm"""
        score = self.evaluate(board,depth)
        
        #terminal state
        if score > 0 or score < 0:
            return score
        if self.is_board_full(board) or depth == 0:
            return 0
        
        if maximizing_player:
            max_eval = float('-inf')
            for r, c in self.get_available_moves(board):
                board[r][c] = self.computer
                
                evaluation = self.minimax(board, depth - 1, False, alpha, beta)
                board[r][c] = EMPTY
                
                max_eval = max(max_eval, evaluation)
                alpha = max(alpha, evaluation)
                
                if beta <= alpha:
                    break 
            return max_eval
            
        else:
            min_eval = float('inf')
            for r, c in self.get_available_moves(board):
                board[r][c] = self.user
                
                evaluation = self.minimax(board, depth - 1, True, alpha, beta)
                board[r][c] = EMPTY
                
                min_eval = min(min_eval, evaluation)
                beta = min(beta, evaluation)
                
                if beta <= alpha:
                    break 
            return min_eval

    def best_move(self):
        """Identifying best move for the computer"""
        best_score = float('-inf')
        move_coords = None
        
        available_moves = self.get_available_moves(self.board)
        depth = len(available_moves)

        for r, c in available_moves:
            self.board[r][c] = self.computer
            score = self.minimax(self.board, depth-1, False, float('-inf'), float('inf'))
            self.board[r][c] = EMPTY
            
            if score > best_score:
                best_score = score
                move_coords = (r, c)
                
        return move_coords

    def play_tic_tac_toe(self):
        """Main game loop implementation. Player1 plays with 'X', player2 plays second with 'O'"""
        print("Welcome to Tic-Tac-Toe!")
        print("Coordinates are 0, 1, or 2 for both rows and columns.")

        while True:
            choice = input("Do you want to play first (1) or second (2)? Enter 1 or 2: ").strip()
            if choice == '1':
                self.user = PLAYER_X
                self.computer = PLAYER_O
                current_turn = 'user'
                break
            elif choice == '2':
                self.user = PLAYER_O
                self.computer = PLAYER_X
                current_turn = 'computer'
                break
            else:
                print("Invalid input. Please enter 1 or 2.")
                
        print(f"\nYou are playing as {self.user}. The computer is {self.computer}.")
        self.print_board()

        #main loop
        while True:
            
            #if someone won
            if self.check_winner(self.board, self.user):
                print("Congratulations! You won!") 
                break
            elif self.check_winner(self.board, self.computer):
                print("Computer wins! Better luck next time.")
                break
            elif self.is_board_full(self.board):
                print("It's a draw!")
                break

            #turn managment
            if current_turn == 'user':
                print("Your turn!")
                while True:
                    try:
                        user_input = input("Enter row and column separated by a space (e.g., '0 1'): ")
                        r, c = map(int, user_input.split())
                        
                        if r not in [0, 1, 2] or c not in [0, 1, 2]:
                            print("Coordinates must be 0, 1, or 2. Try again.")
                        elif self.board[r][c] != EMPTY:
                            print("That cell is already occupied. Try again.")
                        else:
                            self.board[r][c] = self.user
                            break 
                    except ValueError:
                        print("Invalid input format. Please enter two numbers separated by a space.")
                        
                self.print_board()
                current_turn = 'computer'

            else:
                print("Computer is thinking...")
                move = self.best_move()
                if move:
                    self.board[move[0]][move[1]] = self.computer
                self.print_board()
                current_turn = 'user'


if __name__ == "__main__":
    game = TicTacToe()      
    game.play_tic_tac_toe()