import os
import config
import datetime
import numpy as np 
import requests
import json

from bert.tokenization.bert_tokenization import FullTokenizer
from model import BERTModel

import warnings
warnings.filterwarnings("ignore")

'''
To predict the user logs classification result using the saved checkpoint files
'''

class BertPredict:
    def __init__(self,sentence):
        self.sentence = sentence

    def predict(self):
        sent = self.sentence
        classes = ['food','transport','shopping','bills']
        init_time = datetime.datetime.now()
        tokenizer = FullTokenizer(vocab_file=os.path.join(config.BERT_CKPT_DIR,"vocab.txt"))

        pred_tokens = map(tokenizer.tokenize, sent)
        pred_tokens = map(lambda tok: ["[CLS]"] + tok + ["[SEP]"],pred_tokens)
        pred_token_ids = list(map(tokenizer.convert_tokens_to_ids,pred_tokens))

        pred_token_ids = map(lambda tids: tids + [0]*(config.MAX_LEN-len(tids)),pred_token_ids)
        pred_token_ids = np.array(list(pred_token_ids))

        #create the model
        model = BERTModel.create_model(config.MAX_LEN,classes,config.BERT_CKPT_FILE)
        print(model.summary())

        #model prediction using saved model checkpoints
        model.load_weights(config.SAVE_WEIGHTS_PATH)
        print("model loaded")
        predictions = model.predict(pred_token_ids).argmax(axis=-1)
        print(f"prediction: {predictions}")
        print("result")
        final_time = datetime.datetime.now() - init_time
        results_lst = []
        result = dict()
        for text, label in zip(sent,predictions):
            result['text'] = text
            result['label'] = classes[label]
            result['time_taken'] = final_time.total_seconds()
            results_lst.append(result.copy())

        return results_lst

if __name__ == "__main__":
    #testing the predictions
    sentence = ["I purchased a new shoe",
                "i have paid of my car emi",
                "purchased grocery items"]

    pred_res = BertPredict(sentence).predict()
    print(pred_res)

            

