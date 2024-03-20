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
df = pd.read_csv('/cfs/home/u024219/Tese/CARLSy/chess_dataset.csv', sep='|', skipinitialspace= True, encoding_errors='ignore')
#df = pd.read_csv('C:\\Users\\afons\\Ambiente de Trabalho\\dataset\\chess_dataset.csv', sep='|', skipinitialspace= True, encoding_errors='ignore')
df = pd.DataFrame(df)

df['Notation:Commentary'] = df['algebraic_notation'] + ": " + df['commentary']

# Load Dataset from Pandas DataFrame
chess_dataset = Dataset.from_pandas(df)

# Separate Dataset into train, validation and test datasets
chess_dataset = chess_dataset.train_test_split(test_size= 0.1)
chess_dataset_2 = chess_dataset['test'].train_test_split(test_size=0.5)

chess_dataset = DatasetDict({
    'train': chess_dataset['train'],
    'valid': chess_dataset_2['train'],
    'test': chess_dataset_2['test']
})

def get_training_corpus():
    return (
        chess_dataset["train"][i : i + 1000]["Notation:Commentary"]
        for i in range(0, len(chess_dataset["train"]), 1000)
    )

training_corpus = get_training_corpus()

tokenizer = old_tokenizer.train_new_from_iterator(training_corpus, 52000)

tokenizer.save_pretrained("flanT5-finetuned")