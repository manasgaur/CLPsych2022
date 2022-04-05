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
    df = pd.read_csv(path, sep='\t')
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
    # Processing commands
    # --------------
    #df['content'] = df['content'].fillna(' ')
    #df['title'] = df['title'].fillna(' ')
    # ---------------
    return df


class BertDataset:
    def __init__(self,text,target):
        self.tweet = text
        self.target = target
#         self.keyword = keyword
#         self.location = location
        self.tokenizer = TOKENIZER
        self.max_len = TOKENS_MAX_LENGTH
    def __len__(self):
        return len(self.tweet)
    
    def __getitem__(self,item):
        tweet = re.sub(r'http\S+', '', self.tweet[item]) ###removes URL from tweets
        tweet = " ".join(str(tweet).split())
        
        
        inputs = self.tokenizer.encode_plus(
            tweet,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding=True,truncation=True
        )
        
        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]
        
        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask" : torch.tensor(mask, dtype=torch.long),
            "token_type_ids" : torch.tensor(token_type_ids, dtype=torch.long),
            "targets": torch.tensor(self.target[item], dtype=torch.float),
        }   