FROM tensorflow/serving

RUN apt-get update && apt-get install -y --no-install-recommends nginx git


#the bert model should be within this directory
COPY /model/bert model/bert 

COPY nginx.conf /etc/nginx/nginx.conf

ENTRYPOINT service nginx start | tensorflow_model_server --rest_api_port=8501 \
--model_name=bert \
--model_base_path=/model/bert