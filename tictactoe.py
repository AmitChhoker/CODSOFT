import re

# Initialize the board
def initialize_board():
    return [' ' for _ in range(9)]

# Print the board
def print_board(board):
    for i in range(3):
        print(f"{board[3*i]} | {board[3*i+1]} | {board[3*i+2]}")
        if i < 2:
            print("---------")

# Check for a win
def check_win(board, player):
    win_conditions = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Horizontal
        [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Vertical
        [0, 4, 8], [2, 4, 6]               # Diagonal
    ]
    return any(all(board[i] == player for i in condition) for condition in win_conditions)

# Check for a tie
def check_tie(board):
    return all(cell != ' ' for cell in board)

# Minimax algorithm
def minimax(board, depth, is_maximizing):
    if check_win(board, 'O'):
        return 10 - depth
    if check_win(board, 'X'):
        return depth - 10
    if check_tie(board):
        return 0

    if is_maximizing:
        best_score = float('-inf')
        for i in range(9):
            if board[i] == ' ':
                board[i] = 'O'
                score = minimax(board, depth + 1, False)
                board[i] = ' '
                best_score = max(score, best_score)
        return best_score
    else:
        best_score = float('inf')
        for i in range(9):
            if board[i] == ' ':
                board[i] = 'X'
                score = minimax(board, depth + 1, True)
                board[i] = ' '
                best_score = min(score, best_score)
        return best_score

# Find the best move for the AI
def find_best_move(board):
    best_move = -1
    best_score = float('-inf')
    for i in range(9):
        if board[i] == ' ':
            board[i] = 'O'
            score = minimax(board, 0, False)
            board[i] = ' '
            if score > best_score:
                best_score = score
                best_move = i
    return best_move

# Main game function
def play_game():
    board = initialize_board()
    current_player = 'X'  # Player starts first

    while True:
        print_board(board)
        
        if current_player == 'X':
            move = int(input("Enter your move (1-9): ")) - 1
            if 0 <= move < 9 and board[move] == ' ':
                board[move] = 'X'
                if check_win(board, 'X'):
                    print_board(board)
                    print("Congratulations! You win!")
                    break
                current_player = 'O'
            else:
                print("Invalid move. Try again.")
        else:
            move = find_best_move(board)
            board[move] = 'O'
            if check_win(board, 'O'):
                print_board(board)
                print("AI wins!")
                break
            current_player = 'X'

        if check_tie(board):
            print_board(board)
            print("It's a tie!")
            break

if __name__ == "__main__":
    play_game()
