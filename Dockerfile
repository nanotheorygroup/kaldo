# ---- Builder Stage ----
# Start with the slim image to build our dependencies
FROM python:3.12-slim AS builder

WORKDIR /app

# Create a virtual environment to isolate packages
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install them into the venv
COPY requirements.txt .
# install an extra package for parallel runs
RUN pip install --no-cache-dir -r requirements.txt pytest-xdist

# ---- Final Stage ----
# Start fresh from the same slim base image
FROM python:3.12-slim

WORKDIR /app

# Copy the virtual environment from the builder stage
COPY --from=builder /opt/venv /opt/venv

# uncomment to copy the code base
# Since we will load the code from github, no need to copy it
# COPY . .

# Set the PATH to use the python from the venv and run the app
ENV PATH="/opt/venv/bin:$PATH"
CMD ["/bin/bash"]
