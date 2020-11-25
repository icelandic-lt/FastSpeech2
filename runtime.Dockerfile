FROM python:3.7

RUN apt-get update -y && \
  apt-get install python3-pip idle3 libsndfile1-dev -y && \
  pip3 install --no-cache-dir --upgrade pip

WORKDIR /opt/fastspeech2
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
RUN pip3 install --pre torch==1.6.0.dev20200428 -f https://download.pytorch.org/whl/nightly/cu102/torch_nightly.html

COPY . .

ENTRYPOINT ["python3", "./synthesize.py"]
CMD ["--step", "500000"]