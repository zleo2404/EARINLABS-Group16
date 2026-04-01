# Import any required libraries here
EMPTY = ' '
PLAYER_X = 'X'
PLAYER_O = 'O'

class TicTacToe:
    def __init__(self):
        self.board = [[' ' for _ in range(3)] for _ in range(3)]

    def print_board(self):
        for row in self.board:
            print("|".join(row))
            print("-" * 5)
            
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
        
    def evaluate(self, board):
        """
        Evaluation function for the board.
        Returns +10 if the computer wins, -10 if the user wins, and 0 for a draw.
        """
        if self.check_winner(board, self.computer):
            return 10
        elif self.check_winner(board, self.user):
            return -10
        else:
            return 0
        
    def is_board_full(self, board):
        """Checks if there are no empty spaces left on the board."""
        for row in board:
            if ' ' in row:
                return False
        return True
    
    def get_available_moves(self, board):
        """Returns a list of tuples (row, col) for all empty cells."""
        moves = []
        for r in range(3):
            for c in range(3):
                if board[r][c] == ' ':
                    moves.append((r, c))
        return moves

    def minimax(self, board, depth, maximizing_player, alpha, beta):
        """
        Minimax with alpha-beta pruning algorithm
        """
        score = self.evaluate(board)
        
        # Se qualcuno ha vinto
        if score == 10 or score == -10:
            return score
            
        # Se non ci sono mosse disponibili (pareggio)
        if self.is_board_full(board):
            return 0
            
        # Se abbiamo raggiunto il limite di profondità
        if depth == 0:
            return score
        
        # Turno del Computer (Maximizing player)
        if maximizing_player:
            value = float('-inf')
            for move in self.get_available_moves(board):
                row, col = move
                board[row][col] = self.computer
                
                eval = self.minimax(board, depth - 1, False, alpha, beta)
    
                board[row][col] = ' '
                
                value = max(value, eval)
                alpha = max(alpha, value)
                
                if beta <= alpha:
                    break 
            return value
            
        # Turno dell'Utente (Minimizing player)
        else:
            value = float('inf')
            for move in self.get_available_moves(board):
                row, col = move
                board[row][col] = self.user
                
                eval = self.minimax(board, depth - 1, True, alpha, beta)
    
                board[row][col] = ' '
                
                value = min(value, eval)
                beta = min(beta, value)
                
                if beta <= alpha:
                    break 
            return value

    def best_move(self):
        """
        Identifying best move for the computer
        """
        best_score = float('-inf')
        move_coords = None
        
        depth = len(self.get_available_moves(self.board))

        for move in self.get_available_moves(self.board):
            row, col = move
            self.board[row][col] = self.computer
            
            score = self.minimax(self.board, depth, False, float('-inf'), float('inf'))
            
            self.board[row][col] = ' '
            
            if score > best_score:
                best_score = score
                move_coords = (row, col)
                
        return move_coords

    def play_tic_tac_toe(self):
        """
        Main game loop implementation. Player1 plays with 'X', player2 plays second with 'O'
        """
        print("Welcome to Tic-Tac-Toe!")
        print("Coordinates are 0, 1, or 2 for both rows and columns.")

        while True:
            choice = input("Do you want to play first (1) or second (2)? Enter 1 or 2: ").strip()
            if choice == '1':
                self.user = 'X'
                self.computer = 'O'
                current_turn = 'user'
                break
            elif choice == '2':
                self.user = 'O'
                self.computer = 'X'
                current_turn = 'computer'
                break
            else:
                print("Invalid input. Please enter 1 or 2.")
                
        print(f"\nYou are playing as {self.user}. The computer is {self.computer}.")
        self.print_board()

        while True:
            if self.check_winner(self.board, self.user):
                print("Congratulations! You won!") 
                break
            elif self.check_winner(self.board, self.computer):
                print("Computer wins! Better luck next time.")
                break
            elif self.is_board_full(self.board):
                print("It's a draw!")
                break

            if current_turn == 'user':
                print("Your turn!")
                while True:
                    try:
                        user_input = input("Enter row and column separated by a space (e.g., '0 1'): ")
                        r, c = map(int, user_input.split())
                        
                        if r not in [0, 1, 2] or c not in [0, 1, 2]:
                            print("Coordinates must be 0, 1, or 2. Try again.")
                        elif self.board[r][c] != ' ':
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