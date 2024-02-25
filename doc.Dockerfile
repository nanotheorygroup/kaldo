# Use a multi-stage build to keep the final image size down
# Builder stage for installing necessary packages
FROM continuumio/miniconda3 AS builder

# Set working directory
WORKDIR /app

# Copy your application into the container
COPY . /app

# Prevent apt from prompting for input
ENV DEBIAN_FRONTEND=noninteractive

# Update packages and install only the necessary LaTeX packages and cleanup in the same layer to save space
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    texlive-base \
    texlive-latex-base \
    texlive-fonts-recommended \
    texlive-fonts-extra \
    texlive-latex-extra && \
    # Clean up to reduce layer size
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN conda install pip && \
    pip install -r requirements.txt && \
    pip install -r docs/doc_requirements.txt

# Install additional packages needed for your project
RUN conda install conda-forge::pandoc conda-forge::psutil

# Final stage to copy only the necessary files from the builder stage
FROM continuumio/miniconda3

# Copy the app directory from the builder stage to the final image
COPY --from=builder /app /app

# Set the working directory in the final image
WORKDIR /app

# Command to run your application
CMD ["/bin/bash"]
