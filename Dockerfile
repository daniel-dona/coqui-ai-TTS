FROM ubuntu:20.04

RUN apt update
RUN apt upgrade -y

RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections

RUN apt install locales

RUN locale-gen en_US.UTF-8
ENV LANG='en_US.UTF-8' LANGUAGE='en_US:en' LC_ALL='en_US.UTF-8'

RUN apt install build-essential -y 
RUN apt install python3 python3-dev python3-pip -y

RUN mkdir /repo
COPY . /repo
WORKDIR /repo

RUN apt install libsndfile1-dev -y
RUN python3 -m pip install -e .[all]

 
