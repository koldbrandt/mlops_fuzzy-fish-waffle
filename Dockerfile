# Base image 
From python:3.7-slim
ARG WANDB_API_KEY_LOG
ENV WANDB_API_KEY=${WANDB_API_KEY_LOG}
# install python 
RUN apt update && \
apt install --no-install-recommends -y build-essential gcc && \
apt clean && rm -rf /var/lib/apt/lists/* 

RUN  apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    wget \
  && rm -rf /var/lib/apt/lists/*

# install google sdk
RUN wget -nv \
    https://dl.google.com/dl/cloudsdk/release/google-cloud-sdk.tar.gz && \
    mkdir /root/tools && \
    tar xvzf google-cloud-sdk.tar.gz -C /root/tools && \
    rm google-cloud-sdk.tar.gz && \
    /root/tools/google-cloud-sdk/install.sh --usage-reporting=false \
        --path-update=false --bash-completion=false \
        --disable-installation-options && \
    rm -rf /root/.config/* && \
    ln -s /root/.config /config && \
    # Remove the backup directory that gcloud creates
    rm -rf /root/tools/google-cloud-sdk/.install/.backup

ENV PATH $PATH:/root/tools/google-cloud-sdk/bin
RUN echo '[GoogleCompute]\nservice_account = default' > /etc/boto.cfg

COPY requirements.txt requirements.txt
COPY setup.py setup.py
COPY src/ src/
COPY reports/ reports/
COPY models/ models/
COPY conf/ conf/
COPY .dvc/ .dvc/
COPY data.dvc data.dvc

WORKDIR /
RUN pip install -r requirements.txt --no-cache-dir

RUN wandb login $WANDB_API_KEY_LOG

RUN dvc pull

ENTRYPOINT ["python", "-u", "src/models/train_model.py"]