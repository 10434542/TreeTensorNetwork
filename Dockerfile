FROM nvidia/cuda:10.1-cudnn7-devel

RUN apt-get update &&\
    apt-get install -y --no-install-recommends software-properties-common curl gcc &&\
    add-apt-repository -y ppa:deadsnakes/ppa &&\
    apt-get update

RUN apt-get install -y --no-install-recommends python3.8 python3.8-distutils python3.8-dev
WORKDIR /usr/src/app

RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
RUN python3.8 get-pip.py

RUN python3.8 -m pip install torch torchvision

ADD requirements.txt .
RUN python3.8 -m pip install -r requirements.txt

RUN rm requirements.txt
RUN rm get-pip.py
