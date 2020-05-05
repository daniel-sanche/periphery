FROM nvidia/cuda:10.2-base-ubuntu16.04

ENV PATH /opt/conda/bin:$PATH

RUN apt-get update --fix-missing && \
    apt-get install -y wget bzip2 ca-certificates curl git

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-4.5.11-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -tipsy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

ENV PATH /opt/conda/bin:$PATH
ENV LD_LIBRARY_PATH /usr/local/cuda-10.0/lib64:/usr/local/cuda-10.0/extras/CUPTI/lib64:$LD_LIBRARY_PATH

RUN apt-get install -y python3 python-dev python3-dev \
build-essential libssl-dev libffi-dev \
libxml2-dev libxslt1-dev zlib1g-dev \
python-pip


WORKDIR /app

COPY requirements.txt ./

RUN pip install --upgrade pip
RUN pip install --upgrade setuptools

RUN pip install -r requirements.txt

RUN apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender-dev

COPY . /app
ENTRYPOINT python detect.py
