FROM nvidia/cuda:9.2-cudnn7-runtime

RUN apt-get update &&\
    apt-get install -y --no-install-recommends software-properties-common curl gcc &&\
    add-apt-repository -y ppa:deadsnakes/ppa &&\
    apt-get update

RUN apt-get install -y --no-install-recommends python3.8 python3.8-distutils python3.8-dev
WORKDIR /usr/src/app

RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
RUN python3.8 get-pip.py

RUN python3.8 -m pip install --upgrade https://storage.googleapis.com/jax-releases/cuda92/jaxlib-0.1.37-cp38-none-linux_x86_64.whl
RUN python3.8 -m pip install --upgrade jax
ADD requirements.txt .
RUN python3.8 -m pip install -r requirements.txt

RUN rm requirements.txt
RUN rm get-pip.py
