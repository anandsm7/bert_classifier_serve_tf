### BERT classification/serving using tensorflow 

Fine tuning the bert model for user transactions logs classification using tensorflow 2.0



Trained model is then served using the tensorflow serving. 

```
docker run -p 8501:8501 --name tfserving_bert \
--mount type=bind,source=/tmp/bert,target=/models/bert \
-e MODEL_NAME=bert -t tensorflow/serving &
```

BERT served as an API and the results are as follows

```
[{'text': 'I purchased a new shoe', 'label': 'shopping', 'time': 0.155983}, {'text': 'i have paid of my car emi', 'label': 'bills', 'time': 0.155983}, {'text': 'purchased grocery items', 'label': 'food', 'time': 0.155983}, {'text': 'i have filled gas in my car for 100 rupees', 'label': 'transport', 'time': 0.155983}, {'text': 'purchased 1 kg of rice', 'label': 'food', 'time': 0.155983}, {'text': 'purchase the latest earpods', 'label': 'shopping', 'time': 0.155983}]

```

# CREDITS
https://github.com/curiousily/Deep-Learning-For-Hackers/blob/master/18.intent-recognition-with-BERT.ipynb

https://github.com/abhishekkrthakur/mlframework
