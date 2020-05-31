import os 
import config
import numpy as np 
import requests
import json 
import datetime
from bert.tokenization.bert_tokenization import FullTokenizer

'''
Tensorflow serving is used to set up a predict endpoint.Serving is running as a docker container and
to set up the container 

docker run -p 8501:8501 --name tfserving_bert \
--mount type=bind,source=/tmp/bert,target=/models/bert \
-e MODEL_NAME=bert -t tensorflow/serving &

'''


class TFServing:
    def __init__(self,classes,SERVING_URL,sentence):
        self.classes = classes
        self.SERVING_URL = SERVING_URL
        self.sentence = sentence

    def prepare(self):
        sentence = self.sentence
        tokenizer = FullTokenizer(vocab_file=os.path.join(config.BERT_CKPT_DIR,"vocab.txt"))

        pred_tokens = map(tokenizer.tokenize, sentence)
        pred_tokens = map(lambda tok: ["[CLS]"] + tok + ["[SEP]"],pred_tokens)
        pred_token_ids = list(map(tokenizer.convert_tokens_to_ids,pred_tokens))

        pred_token_ids = map(lambda tids: tids + [0]*(config.MAX_LEN-len(tids)),pred_token_ids)
        pred_token_ids = np.array(list(pred_token_ids))
        return pred_token_ids

    def serve(self):
        predict_request = dict()
        preds_dict = dict()
        preds_lst = []

        init_time = datetime.datetime.now()
        pred_token_ids = self.prepare()

        predict_request["instances"] = pred_token_ids.tolist()
        response = requests.post(self.SERVING_URL, data=json.dumps(predict_request))
        final_time = datetime.datetime.now() - init_time
        for i,res in enumerate(response.json()["predictions"]):
            preds_dict["text"] = self.sentence[i]
            preds_dict["label"] = self.classes[np.argmax(res)]
            preds_dict["time"] = final_time.total_seconds()
            preds_lst.append(preds_dict.copy())

        return preds_lst


if __name__ == "__main__":

    SERVING_URL = 'http://localhost:8080/invocations'
    classes = ['shopping','transport','food','bills']
    
    sentences = ["I purchased a new shoe",
                "i have paid of my car emi",
                "purchased grocery items","i have filled gas in my car for 100 rupees","purchased 1 kg of rice",
                "purchase the latest earpods"]

    predict_res = TFServing(classes,SERVING_URL,sentences).serve()

    print(predict_res)

