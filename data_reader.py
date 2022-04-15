"""Read txt/csv file as input."""
import pandas as pd
import numpy as np
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from utils import TOKENIZER,TOKENS_MAX_LENGTH
import torch
import re
import string
import json
from torch.utils.data import Dataset, DataLoader

from utils import FILE_train_users, FOLDER_train_data 


def get_train_users() -> dict():
    """
    Read/process all training data for both tasks &
    return a single dictionary with all of the data.
    
    Parameters
    ----------
    None
        (Reading utils.FILE_train_users & 
        utils.FOLDER_train_data)

    Returns
    -------
    users: dict()
        A dictionary with the following form:
        userid -> 
            {
                timelines:  [list_of_timeline_ids (each being a str)],
                data:       [list_of_timelines (each being pd.DataFrame)],
                risk_level: label_for_task_b (single str)
            }
    """
    users = json.load(open(FILE_train_users, 'rb'))
    for user in users.keys():
        users[user]['data'] = [csv_reader(FOLDER_train_data+str(tlid)+'.tsv') for tlid in users[user]['timelines']]
        if users[user]['risk_level']=='No': #merging the "No Risk/"Low Risk" labels
            users[user]['risk_level'] = 'Low'
    return users


def csv_reader(path: str) -> pd.DataFrame:
    """
    Read & process  a single TSV file.
    
    Parameters
    ----------
    path: str
        path to TSV file.

    Returns
    -------
    df: pd.DataFrame
        dataframe containing a single timeline.
    """
    df = pd.read_csv(path)
    df = process_data(df)

    return df


def process_data(df: pd.DataFrame) -> pd.DataFrame:
    """Process input dataframe.

    Parameters
    ----------    
    df: pd.DataFrame
        Raw dataframe containing timelines.

    Returns
    -------
    df: pd.DataFrame
        processed dataframe
    """
    # Keep str values and fill none values
    if 'title' not in df.columns:
        df['title'] = None
    df['content'] = df['content'].fillna(' ')
    df['title'] = df['title'].fillna(' ')
    df = df[df['title'].apply(lambda x: True if type(x)==str else False)]
    df = df[df['content'].apply(lambda x: True if type(x)==str else False)]
    return df


class BertDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.len = len(dataframe)
        self.data = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __getitem__(self, index):
        title = str(self.data.content[index])
        title = " ".join(title.split())
        inputs = self.tokenizer.encode_plus(
            title,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'targets': torch.tensor(self.data.label[index], dtype=torch.long)
        } 
    
    def __len__(self):
        return self.len
