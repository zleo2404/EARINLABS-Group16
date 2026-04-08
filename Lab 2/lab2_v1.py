EMPTY = ' '
PLAYER_X = 'X'
PLAYER_O = 'O'

class TicTacToe:
    def __init__(self):
        self.board = [[EMPTY for _ in range(3)] for _ in range(3)]
        self.human_symbol = PLAYER_X
        self.computer_symbol = PLAYER_O

    def print_board(self):
        print("  0   1   2")
        for index, row in enumerate(self.board):
            print(index, " | ".join(row))
            if index < 2:
                print("  -----------")

    def check_winner(self, board=None):
        if board is None:
            board = self.board

        for row in board:
            if row[0] != EMPTY and row[0] == row[1] == row[2]:
                return row[0]

        for col in range(3):
            if board[0][col] != EMPTY and board[0][col] == board[1][col] == board[2][col]:
                return board[0][col]

        if board[0][0] != EMPTY and board[0][0] == board[1][1] == board[2][2]:
            return board[0][0]

        if board[0][2] != EMPTY and board[0][2] == board[1][1] == board[2][0]:
            return board[0][2]

        return None

    def is_board_full(self, board=None):
        if board is None:
            board = self.board

        for row in board:
            if EMPTY in row:
                return False
        return True

    def minimax(self, board, depth, maximizing_player, alpha, beta):
        """
        Minimax with alpha-beta pruning algorithm
        Parameters:
        - board: 3x3 matrix, each cell contains either ' ' (empty cell), 'X' (player), or 'O' (computer)
        - depth: depth
        - maximizing_player: boolean which is equal to True when the player tries to maximize the score
        - alpha: alpha variable for pruning
        - beta: beta variable for pruning

        Returns:
        - Best value

        """
        winner = self.check_winner(board)
        if winner == self.computer_symbol:
            return 1
        if winner == self.human_symbol:
            return -1
        if self.is_board_full(board):
            return 0

        if maximizing_player:
            best_value = float("-inf")
            for row in range(3):
                for col in range(3):
                    if board[row][col] == EMPTY:
                        board[row][col] = self.computer_symbol
                        value = self.minimax(board, depth + 1, False, alpha, beta)
                        board[row][col] = EMPTY
                        best_value = max(best_value, value)
                        alpha = max(alpha, best_value)
                        if beta <= alpha:
                            return best_value
            return best_value

        best_value = float("inf")
        for row in range(3):
            for col in range(3):
                if board[row][col] == EMPTY:
                    board[row][col] = self.human_symbol
                    value = self.minimax(board, depth + 1, True, alpha, beta)
                    board[row][col] = EMPTY
                    best_value = min(best_value, value)
                    beta = min(beta, best_value)
                    if beta <= alpha:
                        return best_value
        return best_value


    def best_move(self):
        """
        Identifying best move for the computer
        Parameters:
        - board: 3x3 matrix, each cell contains either ' ' (empty cell), 'X' (player1), or 'O' (player2)

        Returns:
        - best move coordinates
        """
        best_score = float("-inf")
        move = None

        for row in range(3):
            for col in range(3):
                if self.board[row][col] == EMPTY:
                    self.board[row][col] = self.computer_symbol
                    score = self.minimax(self.board, 0, False, float("-inf"), float("inf"))
                    self.board[row][col] = EMPTY

                    if score > best_score:
                        best_score = score
                        move = (row, col)

        return move



    def play_tic_tac_toe(self):
        """
        Main game loop implementation. Player1 plays with 'X', player2 plays second with 'O'

        """
        self.board = [[EMPTY for _ in range(3)] for _ in range(3)]
        print("Welcome to Tic-Tac-Toe!")
        print("Enter moves using row and column numbers from 0 to 2.")
        print("The first player uses 'X' and the second player uses 'O'.")
        print("For example, the top-left cell is (0, 0) and the bottom-right cell is (2, 2).")

        while True:
            choice = input("Do you want to play first or second? Enter 1 for first, 2 for second: ").strip()
            if choice == "1":
                self.human_symbol = PLAYER_X
                self.computer_symbol = PLAYER_O
                human_turn = True
                break
            if choice == "2":
                self.human_symbol = PLAYER_O
                self.computer_symbol = PLAYER_X
                human_turn = False
                break
            print("Invalid choice. Please enter 1 or 2.")

        print(f"You are '{self.human_symbol}'. The computer is '{self.computer_symbol}'.")
        self.print_board()

        while True:
            if human_turn:
                print(f"\nYour turn ({self.human_symbol}):")
                while True:
                    try:
                        row = int(input("Enter row (0-2): "))
                        col = int(input("Enter column (0-2): "))
                    except ValueError:
                        print("Invalid input. Please enter integers from 0 to 2.")
                        continue

                    if row < 0 or row > 2 or col < 0 or col > 2:
                        print("Coordinates out of range. Please enter values from 0 to 2.")
                        continue

                    if self.board[row][col] != EMPTY:
                        print("That cell is already occupied. Choose another one.")
                        continue

                    self.board[row][col] = self.human_symbol
                    print(f"You placed '{self.human_symbol}' at ({row}, {col}).")
                    self.print_board()
                    break
            else:
                print(f"\nComputer's turn ({self.computer_symbol}):")
                move = self.best_move()
                if move is not None:
                    row, col = move
                    self.board[row][col] = self.computer_symbol
                    print(f"Computer placed '{self.computer_symbol}' at ({row}, {col}).")
                    self.print_board()

            winner = self.check_winner()
            if winner is not None:
                if winner == self.human_symbol:
                    print("You win!")
                else:
                    print("Computer wins!")
                break

            if self.is_board_full():
                print("It's a draw!")
                break

            human_turn = not human_turn

if __name__ == "__main__":
    game = TicTacToe()
    game.play_tic_tac_toe()
