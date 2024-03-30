from transformers import AutoTokenizer

from datasets import Dataset, DatasetDict

import pandas as pd

import os

# Select GPUs
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"]= "false"

# Define Old_Tokenizer
old_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")

# Put the dataset in a Pandas DataFrame
df = pd.read_csv('/cfs/home/u024219/Tese/CARLSy/datasets/chess_dataset_extended_with_move.csv', sep='|', skipinitialspace= True, encoding_errors='ignore')
#df = pd.read_csv('C:\\Users\\afons\\Ambiente de Trabalho\\dataset\\chess_dataset.csv', sep='|', skipinitialspace= True, encoding_errors='ignore')
df = pd.DataFrame(df)

df['Training'] ="[PGN]" + df['algebraic_notation'] + ["MOVE"] + df['move'] + "[BOARD]" + df['positions'] + "[ATTACKS] " + df['attacks'] + "[COMMENTARY]" + df['commentary']

# Load Dataset from Pandas DataFrame
chess_dataset = Dataset.from_pandas(df)

def get_training_corpus():
    return (
        chess_dataset[i : i + 1000]["Training"]
        for i in range(0, len(chess_dataset), 1000)
    )

training_corpus = get_training_corpus()

tokenizer = old_tokenizer.train_new_from_iterator(training_corpus, 32128)

tokenizer.save_pretrained("flanT5-finetuned-extended")