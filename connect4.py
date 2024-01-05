import numpy as np
import math
import random
import os
import json
#from tqdm import tqdm
import time
import sys
import threading
from multiprocessing import Pool, cpu_count

ROW_COUNT = 6
COLUMN_COUNT = 7
WINDOW_LENGTH = 4  


PLAYER_PIECE = 1
BOT_PIECE = 2
EMPTY = 0

def create_board():
    board = np.zeros((ROW_COUNT, COLUMN_COUNT))
    return board

def print_board(board):
    flipped_board = np.flipud(board)

    # print white
    print("\033[0;37;47m 0 \033[0;37;47m 1 \033[0;37;47m 2 \033[0;37;47m 3 \033[0;37;47m 4 \033[0;37;47m 5 \033[0;37;47m 6 \033[0m")
    for i in flipped_board:
        row_str = ""

        for j in i:
            if j == 1:
                # Print yellow
                row_str += "\033[0;37;43m 1 "
            elif j == 2:
                # Print red
                row_str += "\033[0;37;41m 2 "
            else:
                # Print navy
                row_str += "\033[0;37;48;5;18m   "
        
        print(row_str + "\033[0m")



def get_valid_locations(board):
    valid_locations = []
    for col in range(COLUMN_COUNT):
        if is_valid(board, col):
            valid_locations.append(col)
    return valid_locations

def drop_piece(board, row, col, piece):
    board[row][col] = piece

def is_valid(board, col):
    return board[ROW_COUNT - 1][col] == 0

def get_next_open_row(board, col):
    for row in range(ROW_COUNT):
        if board[row][col] == 0:
            return row

#def print_board(board):
#    print(np.flip(board, 0))

def score_position(board, piece):
    score = 0

    # centre column
    centre_array = [int(i) for i in list(board[:, COLUMN_COUNT // 2])]
    centre_count = centre_array.count(piece)
    score += centre_count * 3

    #horizontal positions
    for row in range(ROW_COUNT):
        row_array = [int(i) for i in list(board[row, :])]
        for col in range(COLUMN_COUNT - 3):

            window = row_array[col : col + WINDOW_LENGTH]
            score += evaluate_window(window, piece)

    # vertical positions
    for col in range(COLUMN_COUNT):
        col_array = [int(i) for i in list(board[:, col])]
        for row in range(ROW_COUNT - 3):
            window = col_array[row : row + WINDOW_LENGTH]
            score += evaluate_window(window, piece)

    # positive diagonals
    for row in range(ROW_COUNT - 3):
        for col in range(COLUMN_COUNT - 3):
            window = [board[row + i][col + i] for i in range(WINDOW_LENGTH)]
            score += evaluate_window(window, piece)

    # negative diagonals
    for row in range(ROW_COUNT - 3):
        for col in range(COLUMN_COUNT - 3):
            window = [board[row + 3 - i][col + i] for i in range(WINDOW_LENGTH)]
            score += evaluate_window(window, piece)

    return score


def evaluate_window(window, piece):
    score = 0
    
    opp_piece = PLAYER_PIECE
    if piece == PLAYER_PIECE:
        opp_piece = BOT_PIECE

    # winning move
    if window.count(piece) == 4:
        score += 100
    # opponent's winning move
    #elif window.count(opp_piece) == 3 and window.count(EMPTY) == 1:
        #score += 50
    # connecting 3 
    elif window.count(piece) == 3 and window.count(EMPTY) == 1:
        score += 5
    # connecting 2 
    elif window.count(piece) == 2 and window.count(EMPTY) == 2:
        score += 2

    # blocking an opponent's winning move 
    if window.count(opp_piece) == 3 and window.count(EMPTY) == 1:
        score -= 4

    return score

def winning_move(board, piece):
    #horizontal locations
    for c in range(COLUMN_COUNT - 3):
        for r in range(ROW_COUNT):
            if board[r][c] == piece and board[r][c + 1] == piece and board[r][c + 2] == piece and board[r][c + 3] == piece:
                return True

    #vertical locations
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT - 3):
            if board[r][c] == piece and board[r + 1][c] == piece and board[r + 2][c] == piece and board[r + 3][c] == piece:
                return True

    #positive diagonal
    for c in range(COLUMN_COUNT - 3):
        for r in range(ROW_COUNT - 3):
            if board[r][c] == piece and board[r + 1][c + 1] == piece and board[r + 2][c + 2] == piece and board[r + 3][c + 3] == piece:
                return True

    #negative diagonal
    for c in range(COLUMN_COUNT - 3):
        for r in range(3, ROW_COUNT):
            if board[r][c] == piece and board[r - 1][c + 1] == piece and board[r - 2][c + 2] == piece and board[r - 3][c + 3] == piece:
                return True
            

def is_terminal_node(board):
    return winning_move(board, PLAYER_PIECE) or winning_move(board, BOT_PIECE) or len(get_valid_locations(board)) == 0

def minimax(board, depth, alpha, beta, maximisingPlayer):
    valid_locations = get_valid_locations(board)

    is_terminal = is_terminal_node(board)
    if depth == 0 or is_terminal:
        if is_terminal:
            if winning_move(board, BOT_PIECE):
                return (None, 9999999)
            elif winning_move(board, PLAYER_PIECE):
                return (None, -9999999)
            else:  
                return (None, 0)
        else:
            return (None, score_position(board, BOT_PIECE))

    if maximisingPlayer:
        value = -math.inf
        column = random.choice(valid_locations)
        for col in valid_locations:
            row = get_next_open_row(board, col)
            b_copy = board.copy()
            drop_piece(b_copy, row, col, BOT_PIECE)
            new_score = minimax(b_copy, depth - 1, alpha, beta, False)[1]
            if new_score > value:
                value = new_score
                column = col
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return column, value

    else: 
        value = math.inf
        column = random.choice(valid_locations)
        for col in valid_locations:
            row = get_next_open_row(board, col)
            b_copy = board.copy()
            drop_piece(b_copy, row, col, PLAYER_PIECE)
            new_score = minimax(b_copy, depth - 1, alpha, beta, True)[1]
            if new_score < value:
                value = new_score
                column = col
            beta = min(beta, value)
            if alpha >= beta:
                break
        return column, value

def play_manjaro_vs_bot():
    board = create_board()
    print_board(board)
    
    game_over = False
    turn = random.randint(0, 1)  

    while not game_over:
        if turn == 0:
            depth = random.randint(3, 5)  
            col, minimax_score = minimax(board, depth, -math.inf, math.inf, True)
            if is_valid(board, col):
                row = get_next_open_row(board, col)
                drop_piece(board, row, col, BOT_PIECE)
        else:
            depth = random.randint(3, 5)  
            col, minimax_score = minimax(board, depth, -math.inf, math.inf, False)
            if is_valid(board, col):
                row = get_next_open_row(board, col)
                drop_piece(board, row, col, PLAYER_PIECE)
        
        print_board(board)

        if winning_move(board, PLAYER_PIECE):
            print("Player 1 wins!")
            game_over = True
        elif winning_move(board, BOT_PIECE):
            print("Player 2 wins!")
            game_over = True
        elif len(get_valid_locations(board)) == 0:
            print("It's a tie!")
            game_over = True

        turn += 1
        turn = turn % 2


def play_user_vs_bot():
    board = create_board()
    print_board(board)
    
    game_over = False
    turn = random.randint(0, 1) 

    while not game_over:
        if turn == 0:
            while True:
                try:
                    col = int(input("Player make your selection (0-6): "))
                    if col < 0 or col > 6:
                        raise ValueError
                    if is_valid(board, col):
                        row = get_next_open_row(board, col)
                        drop_piece(board, row, col, PLAYER_PIECE)
                        break
                    else:
                        print("Column is full. Choose another column.")
                except ValueError:
                    print("Please enter a valid number between 0 and 6.")

        else:
            col, minimax_score = minimax(board, 4, -math.inf, math.inf, True)
            if is_valid(board, col):
                row = get_next_open_row(board, col)
                drop_piece(board, row, col, BOT_PIECE)
        
        print_board(board)

        if winning_move(board, PLAYER_PIECE):
            print("Player wins!")
            game_over = True
        elif winning_move(board, BOT_PIECE):
            print("Bot wins!")
            game_over = True
        elif len(get_valid_locations(board)) == 0:
            print("It's a tie!")
            game_over = True

        turn += 1
        turn = turn % 2

###
'''
def minimax_worker(args):
    board, depth, alpha, beta, maximisingPlayer, col = args

    valid_locations = get_valid_locations(board)
    is_terminal = is_terminal_node(board)

    if depth == 0 or is_terminal:
        if is_terminal:
            if winning_move(board, BOT_PIECE):
                return col, 9999999 if maximisingPlayer else -9999999
            elif winning_move(board, PLAYER_PIECE):
                return col, -9999999 if maximisingPlayer else 9999999
            else:
                return col, 0

        return col, score_position(board, BOT_PIECE) if maximisingPlayer else score_position(board, PLAYER_PIECE)

    if maximisingPlayer:
        value = -9999999
        for col in valid_locations:
            row = get_next_open_row(board, col)
            b_copy = board.copy()
            drop_piece(b_copy, row, col, BOT_PIECE)
            new_col, new_score = minimax_worker((b_copy, depth - 1, alpha, beta, False, col))
            if new_score > value:
                value = new_score
                alpha = max(alpha, value)
                best_col = col
            if alpha >= beta:
                break
        return best_col, value

    else:
        value = 9999999
        for col in valid_locations:
            row = get_next_open_row(board, col)
            b_copy = board.copy()
            drop_piece(b_copy, row, col, PLAYER_PIECE)
            new_col, new_score = minimax_worker((b_copy, depth - 1, alpha, beta, True, col))
            if new_score < value:
                value = new_score
                beta = min(beta, value)
                best_col = col
            if alpha >= beta:
                break
        return best_col, value

def minimax(board, depth, alpha, beta, maximisingPlayer):
    valid_locations = get_valid_locations(board)
    args_list = []

    for col in valid_locations:
        row = get_next_open_row(board, col)
        b_copy = board.copy()
        drop_piece(b_copy, row, col, BOT_PIECE if maximisingPlayer else PLAYER_PIECE)
        args_list.append((b_copy, depth - 1, alpha, beta, not maximisingPlayer, col))

    pool = Pool(cpu_count())
    results = pool.map(minimax_worker, args_list)
    pool.close()
    pool.join()

    if maximisingPlayer:
        best_value = -math.inf
        for col, value in results:
            if value > best_value:
                best_value = value
                best_col = col
        return best_col, best_value
    else:
        best_value = math.inf
        for col, value in results:
            if value < best_value:
                best_value = value
                best_col = col
        return best_col, best_value
'''



def collect_statistics(num_games):
    bot_wins = 0
    player_wins = 0
    ties = 0

    for _ in range(num_games):
        board = create_board()  
        game_over = False
        turn = random.randint(0, 1)

        while not game_over:
            if turn == 0:
                col, minimax_score = minimax(board, 4, -math.inf, math.inf, True)
                if is_valid(board, col):
                    row = get_next_open_row(board, col)
                    drop_piece(board, row, col, BOT_PIECE)
            else:
                col, minimax_score = minimax(board, 4, -math.inf, math.inf, False)
                if is_valid(board, col):
                    row = get_next_open_row(board, col)
                    drop_piece(board, row, col, PLAYER_PIECE)

            if winning_move(board, PLAYER_PIECE):
                player_wins += 1
                game_over = True
            elif winning_move(board, BOT_PIECE):
                bot_wins += 1
                game_over = True
            elif len(get_valid_locations(board)) == 0:
                ties += 1
                game_over = True

            turn += 1
            turn = turn % 2

    total_games = bot_wins + player_wins + ties
    bot_win_rate = (bot_wins / total_games) * 100
    player_win_rate = (player_wins / total_games) * 100
    tie_rate = (ties / total_games) * 100

    statistics = {
        "total_games": total_games,
        "bot_win_rate": bot_win_rate,
        "player_win_rate": player_win_rate,
        "tie_rate": tie_rate
    }

    return statistics

def save_statistics(new_statistics):
    filepath = 'statistics.json'
    all_statistics = []

    try:
        with open(filepath, 'r') as file:
            all_statistics = json.load(file)
    except FileNotFoundError:
        pass  

    all_statistics.extend(new_statistics)

    with open(filepath, 'w') as file:
        json.dump(all_statistics, file, indent=4)


def indeterminate_progress(done):
    while not done.is_set():
        for symbol in '|/-\\':
            sys.stdout.write('\rGathering statistics ' + symbol)
            sys.stdout.flush()
            time.sleep(0.1)

def main():
    all_statistics = []  

    while True:
        print("Welcome to Connect Four!")
        print("Select an option:")
        print("1. Play Connect Four")
        print("2. Display Game Statistics")
        print("3. Exit")

        while True:
            try:
                choice = int(input("Enter your choice (1, 2, or 3): "))
                if choice not in [1, 2, 3]:
                    raise ValueError
                break
            except ValueError:
                print("Please enter a valid choice (1, 2, or 3).")

        if choice == 1:
            print("Choose an option:")
            print("1. Play against the bot")
            print("2. Watch bot vs bot")

            while True:
                try:
                    sub_choice = int(input("Enter your choice (1 or 2): "))
                    if sub_choice not in [1, 2]:
                        raise ValueError
                    break
                except ValueError:
                    print("Please enter a valid choice (1 or 2).")

            if sub_choice == 1:
                play_user_vs_bot()
            else:
                play_manjaro_vs_bot()

        elif choice == 2:
            done = threading.Event()
            spinner_thread = threading.Thread(target=indeterminate_progress, args=(done,))
            spinner_thread.start()
            num_games = 5 
            statistics = collect_statistics(num_games)
            done.set() 
            spinner_thread.join() 

            print("\nStatistics:")
            print(statistics)
            all_statistics.append(statistics)
            save_statistics(all_statistics)

        else:
            print("Exiting the game. Goodbye!")
            save_statistics(all_statistics)  
            break

        replay = input("Do you want to play again? (yes/no): ")
        if replay.lower() != 'yes' and replay.lower() != 'y':
            save_statistics(all_statistics)  
            break

if __name__ == "__main__":
    main()
