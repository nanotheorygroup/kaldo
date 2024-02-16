FROM continuumio/miniconda3

WORKDIR /app
COPY . /app

RUN conda install pip \
    && conda install -c conda-forge pandoc \
    && conda install make \
    && conda install psutil>=5.7.2 \
    && pip install -r /app/requirements.txt \
    && pip install pytest \
    && pip install -r requirements.txt

RUN python -m pytest -v --cov=kaldo --cov-report=xml --color=yes kaldo/tests/
