FROM continuumio/miniconda3

WORKDIR /app
COPY . /app

RUN apt-get update && apt-get upgrade -y && \
    apt-get --no-install-recommends -y install locales-all libstdc++6 && \
    dpkg-reconfigure locales-all

RUN apt-get update && apt-get --no-install-recommends -y install texlive-full

RUN apt-get -y autoremove && \
    apt-get autoclean && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN conda install pip && \
    pip install -r requirements.txt && \
    pip install -r docs/doc_requirements.txt

RUN conda install conda-forge::pandoc && \
    conda install conda-forge::psutil

CMD ["/bin/bash"]
