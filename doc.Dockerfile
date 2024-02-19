FROM continuumio/miniconda3

WORKDIR /app
COPY . /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    texlive \
    ghostscript \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN conda install pip \
    && pip install -r requirements.txt \
    && pip install -r docs/doc_requirements.txt

RUN conda install conda-forge::pandoc
RUN conda install conda-forge::psutil

CMD ["/bin/bash"]
