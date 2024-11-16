import csv
import os
import chess.pgn
from langdetect import detect
from langdetect import DetectorFactory
from langdetect import LangDetectException

DetectorFactory.seed = 0

# Path for folder that contains the PGN files
directory = 

field = ["id", "algebraic_notation", "move", "positions", "attacks", "commentary", "length"]

commentary_sizes = []
commentary_sizes_cleansed = []

algebraic_notation_with_move_sizes = []
algebraic_notation_sizes = []
algebraic_notation_sizes_cleansed = []

positions_sizes = []
positions_sizes_cleansed = []

attacks_sizes = []
attacks_sizes_cleansed = []

move_sizes = []
move_sizes_cleansed = []

id_ = 1
file_number = 0

# Find squares of pieces and attacks in the position
def positions_attacks(board):
    white_pos = "White "
    black_pos = "Black "

    white_atk = "White "
    black_atk = "Black "

    for row in range(0,8):
        for col in range(0,8):
            squareIndex=row*8+col
            square=chess.SQUARES[squareIndex]
            piece = board.piece_at(square)

            if piece != None:

                attacks = board.attacks(square)

                for attack in attacks:
                    if piece.color == chess.WHITE and board.piece_at(attack) != None and board.piece_at(attack).color == chess.BLACK:
                        white_atk += str(piece) + "_" + chess.square_name(square) + "$" + str(board.piece_at(attack)) + "_" + chess.square_name(attack) + " "
                    elif piece.color == chess.BLACK and board.piece_at(attack) != None and board.piece_at(attack).color == chess.WHITE:
                        black_atk += str(piece) + "_" + chess.square_name(square) + "$" + str(board.piece_at(attack)) + "_" + chess.square_name(attack) + " "

                if piece.color == chess.WHITE:
                    white_pos += str(piece) + "_" + chess.square_name(square) + " "
                else:
                    black_pos += str(piece) + "_" + chess.square_name(square) + " "

    return white_pos + black_pos, white_atk + black_atk

#os. remove('C:\\Users\\afons\\Ambiente de Trabalho\\dataset\\chess_dataset_cleanse.csv')
with open('C:\\Users\\afons\\Ambiente de Trabalho\\dataset\\chess_dataset_final.csv', 'w', newline='') as file:
    dataset = csv.writer(file, delimiter='|')
    
    dataset.writerow(field)

    for filename in os.listdir(directory):
        file_number += 1
        games = os.path.join(directory, filename)

        game = open(games, "r", errors="ignore")

        pgn = open(games, errors="ignore")

        chess_game = chess.pgn.read_game(pgn)
        moves = []
        board = chess_game.board()
        
        for move in chess_game.mainline_moves():
            moves.append(move)
            board.push(move)
        board = chess_game.board()
       
        move_number = 0
    
        # Parsing
        notation = ""
        commentary = ""
        while True:
            char = game.read(1) 
            if not char: 
                break
            
            # Skip Headers
            elif char == '[':
                extra = 0
                while True:
                    char = game.read(1) 

                    # Inside loop handling
                    if char =='[':
                        extra +=1
                    if char == ']':
                        extra-=1

                    if extra == -1:
                        # Reset notation and commentary after leaving the headers
                        notation = ""
                        commentary = ""
                        break
            
            # Save Commentary
            elif char == '{':

                # If notation is empty, we skip the pre-game commentaries
                extra = 0
                if notation == "":
                    while True:
                        char = game.read(1) 
                        if char == '{': 
                            extra+=1
                        if char == '}': 
                            extra-=1
                        if extra == -1: 
                            break
                else:
                    while True:
                        char = game.read(1) 

                        # Inside loop handling
                        if char == '{': 
                            extra+=1
                        if char == '}': 
                            extra-=1

                        # If the comment is finished, writerow and reset commentary
                        if extra == -1:
                            pos, attack = positions_attacks(board)
                            move_n = move_number// 2
                            notation_2 = notation
                            if len(notation_2) > 1:
                                while notation_2[-2] != " ":
                                    notation_2 = notation_2[:-1]
                                notation_2 = notation_2[:-1]
                            else:
                                notation_2 = ""
                            if (move_number % 2 == 0):
                                move = str(move_n) + "... "  
                            else:
                                move_n += 1
                                move = str(move_n) + ". "  
                                while True:
                                    if len(notation_2) > 1 and notation_2[-2] != " ":
                                        notation_2 = notation_2[:-1]
                                    else:
                                        notation_2 = notation_2[:-1]
                                        break
                            board.pop()
                            move += board.san(moves[move_number-1])
                            board.push(moves[move_number-1])

                            # Save Commentary if bigger than 15 characters and if it is in english
                            if (len(commentary) > 15):
                                try:
                                    lang = detect(commentary)

                                # If no language detected skip the data point
                                except LangDetectException:
                                    pass


                                if (lang == 'en'):
                                    commentary_sizes_cleansed.append(len(commentary))
                                    algebraic_notation_sizes_cleansed.append(len(notation_2))
                                    positions_sizes_cleansed.append(len(pos))
                                    attacks_sizes_cleansed.append(len(attack))
                                    move_sizes_cleansed.append(len(move))

                                    if (len(commentary) < 61):
                                        dataset.writerow([id_,notation_2, move, pos, attack, commentary.lower(), "[SMALL]"])

                                    elif (len(commentary) < 156):
                                        dataset.writerow([id_,notation_2, move, pos, attack, commentary.lower(), "[MEDIUM]"])

                                    else:
                                        dataset.writerow([id_,notation_2, move, pos, attack, commentary.lower(), "[LARGE]"])

                                    
                                    id_ += 1

                            commentary_sizes.append(len(commentary))
                            algebraic_notation_with_move_sizes.append(len(notation))
                            algebraic_notation_sizes.append(len(notation_2))
                            positions_sizes.append(len(pos))
                            attacks_sizes.append(len(attack))
                            move_sizes.append(len(move))
                            
                            commentary = ""
                            break

                        elif char == ';':
                            commentary += "," 
                        elif char == '\n':
                            commentary += " "
                        elif char == ' ':
                            if len(commentary)>0:
                                commentary += ' '
                        else:
                            commentary += char

            else:
                if char == ' ':
                    if len(notation) > 0:
                        if notation[-1] != ' ':
                            notation += ' '

                elif char == "\n":
                    if len(notation) > 0:
                        notation += ' '

                else:
                    if len(notation) > 1:
                        if (notation[-1] == " " and char.isalpha()):
                            board.push(moves[move_number])
                            move_number += 1
                    notation += char
                
                # Remove the ellipsis placed when it is white's turn to move
                if char == ".":
                    if len(notation) > 3:
                        if notation[-2] == ".":
                            if notation[-3] == ".":
                                #print(notation)
                                while (len(notation) > 1 and notation[-1] != ' '):
                                    notation = notation[:-1]

        if (file_number%1000 == 0):
            print("Finished file " + str(file_number) + " of 12769.")
        game.close()    

from statistics import mean 

print("Max Commentary Size: " + str(max(commentary_sizes)))
print("Min Commentary Size: " + str(min(commentary_sizes)))
print("Mean Commentary Size: " + str(mean(commentary_sizes)))

print("Max Commentary Size Cleansed: " + str(max(commentary_sizes_cleansed)))
print("Min Commentary Size Cleansed: " + str(min(commentary_sizes_cleansed)))
print("Mean Commentary Size Cleansed: " + str(mean(commentary_sizes_cleansed)))

print("Max Algebraic Notation With Move Size: " + str(max(algebraic_notation_with_move_sizes)))
print("Min Algebraic Notation With Move Size: " + str(min(algebraic_notation_with_move_sizes)))
print("Mean Algebraic Notation With Move Size: " + str(mean(algebraic_notation_with_move_sizes)))

print("Max Algebraic Notation Size: " + str(max(algebraic_notation_sizes)))
print("Min Algebraic Notation Size: " + str(min(algebraic_notation_sizes)))
print("Mean Algebraic Notation Size: " + str(mean(algebraic_notation_sizes)))

print("Max Algebraic Notation Size Cleansed: " + str(max(algebraic_notation_sizes_cleansed)))
print("Min Algebraic Notation Size Cleansed: " + str(min(algebraic_notation_sizes_cleansed)))
print("Mean Algebraic Notation Size Cleansed: " + str(mean(algebraic_notation_sizes_cleansed)))

print("Max Position Size: " + str(max(positions_sizes)))
print("Min Position Size: " + str(min(positions_sizes)))
print("Mean Position Size: " + str(mean(positions_sizes)))

print("Max Position Size Cleansed: " + str(max(positions_sizes_cleansed)))
print("Min Position Size Cleansed: " + str(min(positions_sizes_cleansed)))
print("Mean Position Size Cleansed: " + str(mean(positions_sizes_cleansed)))

print("Max Attacks Size: " + str(max(attacks_sizes)))
print("Min Attacks Size: " + str(min(attacks_sizes)))
print("Mean Attacks Size: " + str(mean(attacks_sizes)))

print("Max Attacks Size Cleansed: " + str(max(attacks_sizes_cleansed)))
print("Min Attacks Size Cleansed: " + str(min(attacks_sizes_cleansed)))
print("Mean Attacks Size Cleansed: " + str(mean(attacks_sizes_cleansed)))

print("Max Move Size: " + str(max(move_sizes)))
print("Min Move Size: " + str(min(move_sizes)))
print("Mean Move Size: " + str(mean(move_sizes)))

print("Max Move Size Cleansed: " + str(max(move_sizes_cleansed)))
print("Min Move Size Cleansed: " + str(min(move_sizes_cleansed)))
print("Mean Move Size Cleansed: " + str(mean(move_sizes_cleansed)))


commentary_sizes.sort()
print("Commentary First Tertile Length: " + str(commentary_sizes[116678]))
print("Commentary Second Tertile Length: " + str(commentary_sizes[233357]))

commentary_sizes_cleansed.sort()
print("Commentary First Tertile Length Cleansed: " + str(commentary_sizes_cleansed[100997]))
print("Commentary Second Tertile Length Cleansed: " + str(commentary_sizes_cleansed[201995]))
