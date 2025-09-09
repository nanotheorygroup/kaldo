# Use a multi-stage build to keep the final image size down
# Builder stage for installing necessary packages
FROM continuumio/miniconda3

# Set working directory
WORKDIR /app

# Copy your application into the container
COPY . /app

# Prevent apt from prompting for input
ENV DEBIAN_FRONTEND=noninteractive

# Install Python dependencies
RUN conda install pip && \
    conda install conda-forge::psutil && \
    pip install -r docs/doc_requirements.txt

# Install additional packages needed for your project
RUN conda install conda-forge::pandoc && \
    conda install anaconda::make

# Clean out python cache
RUN pip cache purge
RUN conda clean --all

# Set the working directory in the final image
WORKDIR /app

# Command to run your application
CMD ["/bin/bash"]
