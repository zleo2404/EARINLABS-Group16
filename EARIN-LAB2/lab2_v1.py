# Import any required libraries here
EMPTY = ' '
PLAYER_X = 'X'
PLAYER_O = 'O'

class TicTacToe:
    def __init__(self):
        self.board = board = [[' ' for _ in range(3)] for _ in range(3)]

    def print_board(self):
        for row in board:
            print("|".join(row))
            print("-" * 9)
            

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
        #Your code starts here 


    def best_move(self):
        """
        Identifying best move for the computer
        
        Parameters:
        - board: 3x3 matrix, each cell contains either ' ' (empty cell), 'X' (player1), or 'O' (player2)

        Returns:
        - best move coordinates
        """
        #Your code starts here



    def play_tic_tac_toe(self):
        """
        Main game loop implementation. Player1 plays with 'X', player2 plays second with 'O'

        """
        print("Welcome to Tic-Tac-Toe!")

        while True:
            #Your code starts here

if __name__ == "__main__":
    play_tic_tac_toe()
