import os
import datetime

import config
from dataset import DatasetLoader
from dataset import IntentProcessor
from model import BERTModel

from tensorflow import keras
import matplotlib.pyplot as plt

import pandas as pd 
import numpy as np 

from bert.tokenization.bert_tokenization import FullTokenizer

import warnings
warnings.filterwarnings("ignore")



def run():

    #load the traing data
    df = pd.read_csv(config.TRAINING_FILE)
    #initial preprocessing
    train, test = DatasetLoader(df).get_input()
    classes = df.cat.unique().tolist()
    tokenizer = FullTokenizer(vocab_file=os.path.join(config.BERT_CKPT_DIR,"vocab.txt"))
    #preparing the text for BERT classifier
    data = IntentProcessor(train, test, tokenizer, classes, max_seq_len=128)
    #initiate the BERT model
    model = BERTModel.create_model(config.MAX_LEN,classes,config.BERT_CKPT_FILE)
    print(model.summary())

    #model training
    model.compile(optimizer = keras.optimizers.Adam(1e-5),
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics = [keras.metrics.SparseCategoricalAccuracy(name="acc")])

    log_dir = config.OUTPUT_PATH + "/intent_classifier/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%s")
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir)


    history = model.fit(
        x = data.train_x,
        y = data.train_y,
        validation_split=0.1,
        batch_size = config.BATCH_SIZE,
        shuffle=True,
        epochs = config.EPOCHS,
        callbacks = [tensorboard_callback]
    )

    #saving model checkpoint
    model.save_weights(config.SAVE_WEIGHTS_PATH)

    #plotting accuracy and loss
    plt.title("Accuracy")
    plt.plot(history.history["acc"],label="acc")
    plt.plot(history.history["val_acc"],label="val_acc")
    plt.legend()
    plt.show()

    plt.title("Loss")
    plt.plot(history.history["loss"],label="loss")
    plt.plot(history.history["val_loss"],label="val_loss")
    plt.legend()
    plt.show()


if __name__=="__main__":
    run()