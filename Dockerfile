FROM python:3.9.2-buster

RUN apt-get update

RUN apt-get upgrade -y

RUN mkdir /semantic-emb-api

WORKDIR /semantic-emb-api

COPY ./requirements.txt /semantic-emb-api/requirements.txt

RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

RUN python -m pip install -r requirements.txt

COPY . /semantic-emb-api

RUN python setup.py
