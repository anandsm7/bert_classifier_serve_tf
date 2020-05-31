import config

import numpy as np 
import pandas as pd 

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf  
from tensorflow import keras

import bert 
from bert import BertModelLayer
from bert.loader import StockBertConfig, map_stock_config_to_params,load_stock_weights
from bert.tokenization.bert_tokenization import FullTokenizer

import warnings
warnings.filterwarnings("ignore")



class BERTModel:
    def __init__(self):
        return "BERT MODEL"

    def create_model(max_seq_len, classes, bert_ckpt_file):

        with tf.io.gfile.GFile(config.BERT_CONFIG_FILE, "r") as reader:
            bc = StockBertConfig.from_json_string(reader.read())
            bert_params = map_stock_config_to_params(bc)
            bert_params.adapter_size = None
            bert = BertModelLayer.from_params(bert_params,name='bert')
        
        input_ids = keras.layers.Input(shape = (max_seq_len,), 
                                        dtype='int32',name="input_ids")
        bert_output = bert(input_ids)

        print(f"Shape of BERT Embedding layer :{bert_output.shape}")
        #input will be having a shape of (None,max_seq_len,hidden_layer(768))
        #we can use lambda function to reshape it to (None,hidden_layer)
        cls_out = keras.layers.Lambda(lambda seq:seq[:,0,:])(bert_output)
        cls_out = keras.layers.Dropout(0.5)(cls_out)
        dense = keras.layers.Dense(units=768,activation="tanh")(cls_out)
        dropout = keras.layers.Dropout(0.5)(dense)
        output = keras.layers.Dense(units=len(classes), activation="softmax")(dropout)

        model = keras.Model(inputs=input_ids, outputs=output)
        model.build(input_shape=(None, max_seq_len))

        load_stock_weights(bert, bert_ckpt_file)

        return model