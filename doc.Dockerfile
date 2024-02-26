# Use a multi-stage build to keep the final image size down
# Builder stage for installing necessary packages
FROM continuumio/miniconda3

# Set working directory
WORKDIR /app

# Copy your application into the container
COPY . /app

# Prevent apt from prompting for input
ENV DEBIAN_FRONTEND=noninteractive

# Update packages and install only the necessary LaTeX packages and cleanup in the same layer to save space
RUN apt-get update && \
    apt-get upgrade -y

RUN apt-get install -y --no-install-recommends texlive-base \
    texlive-latex-base \
    texlive-fonts-recommended \
    texlive-fonts-extra \
    texlive-latex-extra \
    dvipng && \
    # Clean up to reduce layer size
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN conda install pip && \
    conda install conda-forge::psutil && \
    pip install -r requirements.txt && \
    pip install -r docs/doc_requirements.txt

# Install additional packages needed for your project
RUN conda install conda-forge::pandoc && \
    conda install anaconda::make

# Set the working directory in the final image
WORKDIR /app

# Command to run your application
CMD ["/bin/bash"]
