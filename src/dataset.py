import re 
import os
import math
import datetime
from tqdm import tqdm

from bert.tokenization.bert_tokenization import FullTokenizer

import numpy as np 
import pandas as pd 

import config

class DatasetLoader:
    def __init__(self,df):
        self.df = df
    
    def __len__(self):
        return self.df.shape

    def cleaner(self,text):
        text = text.replace("'d"," would")
        text = text.replace("'t"," not")
        text = text.replace("'s"," is")
        text = re.sub(r'[^A-Za-z0-9!?]'," ", text)
        text = re.sub(r' +', ' ',text)

        return text
    
    def get_input(self):
        self.df.drop_duplicates(subset="logs",
                   keep="first",
                   inplace=True)
        self.df.reset_index(drop=True)
        self.df["logs"] = self.df["logs"].apply(lambda x: self.cleaner(x))
        self.df = self.df.sample(frac = 1)

        #saving the preprocessed text data
        #self.df.to_csv(config.RESULT_CSV)
        
        train = self.df[self.df.shape[0]//config.SPLIT:]
        test = self.df[:self.df.shape[0]//config.SPLIT]

        return train, test
    

class IntentProcessor:

  DATA_COLUMN = "logs"
  LABEL_COLUMN = "cat"

  def __init__(self,
               train,test,
               tokenizer: FullTokenizer,
               classes,
               max_seq_len = config.MAX_SEQ_LEN
               ):
    self.tokenizer = tokenizer
    self.max_seq_len = 0
    self.classes = classes

    train, test = map(lambda x:x.reindex(x[IntentProcessor.DATA_COLUMN].str.len().sort_values().index),[train,test])
    ((self.train_x, self.train_y), (self.test_x,self.test_y)) = map(self._prepare,[train,test])
    self.max_seq_len = min(self.max_seq_len, max_seq_len)
    self.train_x, self.test_x = map(self._pad, [self.train_x, self.test_x])

  def _prepare(self, df_x):
    x, y = [], []
    for _, row in tqdm(df_x.iterrows()):
      text, label = row[IntentProcessor.DATA_COLUMN], row[IntentProcessor.LABEL_COLUMN]

      tokens = self.tokenizer.tokenize(text)
      tokens = ["[CLS]"] + tokens + ["[SEP]"]
      token_ids = self.tokenizer.convert_tokens_to_ids(tokens)

      self.max_seq_len = max(self.max_seq_len, len(token_ids))
      x.append(token_ids)
      y.append(self.classes.index(label))
    
    return np.array(x), np.array(y)

  def _pad(self, ids):
    x = []
    for input_ids in ids:
      input_ids = input_ids[:min(len(input_ids), self.max_seq_len-2)]
      input_ids = input_ids + [0] * (self.max_seq_len - len(input_ids))
      x.append(np.array(input_ids))
    return np.array(x)



        


