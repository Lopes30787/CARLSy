import csv
import os

directory = "C:\\Users\\afons\\Ambiente de Trabalho\\pgn_parser\\gameknot"
field = ["id", "algebraic_notation", "commentary"]

id_ = 1
file_number = 0

os. remove('C:\\Users\\afons\\Ambiente de Trabalho\\pgn_parser\\chess_dataset.csv')
with open('C:\\Users\\afons\\Ambiente de Trabalho\\pgn_parser\\chess_dataset.csv', 'w', newline='') as file:
    dataset = csv.writer(file, delimiter='|')
    
    dataset.writerow(field)

    for filename in os.listdir(directory):
        file_number += 1
        games = os.path.join(directory, filename)

        game = open(games, "r", errors="ignore")
        
        #Parsing
        notation = ""
        commentary = ""
        while True:
            char = game.read(1) 
            if not char: 
                break
            
            #Skip Headers
            if char == '[':
                while True:
                    char = game.read(1) 
                    if char == ']':
                        #Reset notation and commentary after leaving the headers
                        notation = ""
                        commentary = ""
                        break
            
            #Save Commentary
            if char == '{':
                #If notation is empty, we skip the pre-game commentaries
                if notation == "":
                    while True:
                        char2 = game.read(1) 
                        if char2 == '}': 
                            break
                else:
                    while True:
                        char2 = game.read(1) 
                        #If the comment is finished, writerow and reset commentary
                        if char2 == '}': 
                            dataset.writerow([id_,notation, commentary])
                            commentary = ""
                            id_ += 1
                            break

                        if char2 == ';':
                            commentary += "," 
                        if char2 == '\n':
                            commentary += " "
                        else:
                            commentary += char2

            else:
                if char == " ":
                    if len(notation) > 1:
                        if notation[-1] != " ":
                            notation += " "

                elif char != "\n":
                    notation += char

                else:
                    notation += " "
                
                if char == ".":
                    if len(notation) > 3:
                        if notation[-2] == ".":
                            if notation[-3] == ".":
                                #print(notation)
                                while notation[-1] != " ":
                                    notation = notation[:-1]

        print("Finished file " + str(file_number) + " of 12769.")
        game.close()    

        
        


