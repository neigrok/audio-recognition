FROM ubuntu:18.04
RUN apt-get update
RUN apt-get install -y git curl vim python3-dev python3-pip ffmpeg

RUN pip3 install tensorflow && \
    pip3 install numpy pandas sklearn matplotlib librosa jupyter && \
    pip3 install keras 

RUN mkdir projects
RUN cd projects
RUN mkdir data
RUN git clone https://github.com/neigrok/audio-recognition.git

RUN echo "jupyter notebook --ip 0.0.0.0 --allow-root  --no-browser" > run_jupiter.sh

ENTRYPOINT /bin/bash
