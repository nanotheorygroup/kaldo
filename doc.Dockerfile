# Start from a Debian Slim-based image
FROM python:3.12-slim-bookworm

# Set an environment variable to ensure apt-get runs non-interactively
ENV DEBIAN_FRONTEND=noninteractive

# Update package lists, install pandoc and make, and clean up in a single layer
RUN apt-get update && \
    apt-get install -y --no-install-recommends pandoc make && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# install python packages
COPY docs/doc_requirements.txt /tmp/doc_requirements.txt
RUN pip install --no-cache-dir -r /tmp/doc_requirements.txt && \
    pip cache purge

# uncomment to copy the code base
# Since we will load the code from github, no need to copy it
# COPY . .

# Set workdir
WORKDIR /app

CMD ["/bin/bash"]
