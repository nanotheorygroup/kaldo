FROM continuumio/miniconda3

WORKDIR /app

COPY requirements.txt /app/

RUN conda install -y python=3.10 pip make psutil>=5.7.2 \
    && conda install -y -c conda-forge pandoc curl \
    && pip install --no-cache-dir -r /app/requirements.txt \
    && pip install --no-cache-dir pytest pytest-cov 'pytest-xdist[psutil]'

COPY . /app

CMD ["/bin/bash"]
