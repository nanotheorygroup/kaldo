
# Dockerfiles

## Overview
These Docker container are designed for computational simulations, integrating LAMMPS with Python support and kALDo software. Tailored for CPU environments, it also provides instructions for GPU usage adaptation.

## Getting Started

### CPU Build

To configure the Docker environment for CPU-based computations, rename the CPU-specific Dockerfile:

```bash
mv Dockerfile-cpu Dockerfile
```

#### Building the Container

Build the Docker image with the following command:

```bash
docker build -t kaldo .
```

#### Running the Container

Execute the following to run KALDo in an interactive shell environment:

```bash
docker run -it --rm -u $(id -u):$(id -g) kaldo /bin/bash
```

This grants access to a full Ubuntu environment within the container.

##### File Transfer Support

For easy file transfer, mount your current directory inside the container using the `-v` option:

```bash
docker run -it --rm -u $(id -u):$(id -g) -v $(pwd):/root/ kaldo /bin/bash
```

This command mounts your present working directory to `/root` inside the container.

### GPU Build

To enable GPU support, follow these instructions, ensuring you have `nvidia-cuda-toolkit` and compatible drivers installed.

#### Preparing the Dockerfile

Switch to the GPU-specific Dockerfile:

```bash
mv Dockerfile-gpu Dockerfile
```

#### Building the Container

Build the GPU-enabled Docker image with:

```bash
docker build -t kaldo-gpu .
```

#### Running the Container

For container execution with GPU support:

```bash
docker run -it --rm -u $(id -u):$(id -g) --gpus all kaldo-gpu /bin/bash
```

Use the `--gpus all` flag to enable all GPUs or `--gpus=0,1` for specific GPUs.

### Development and Testing

Set up kALDo for testing or contributions with the Docker-dev environment.

To run development tests with pytest, execute:

```bash
mv Dockerfile-dev Dockerfile
docker build --tag kaldo-dev .
docker run --rm --gpus all kaldo-dev pytest
```

Post-execution, verify kALDo's GPU support and additional features. The container installs the main branch of kALDo, suitable for a test environment.

### Additional Resources

- For detailed Docker and GPU support information, consult [here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) and [here](https://www.tensorflow.org/install/docker).
- The development version (`devel`) includes the full kALDo package, intended for debugging or full access to kALDo's features.
- DeepMD base kit can be found at https://hub.docker.com/r/deepmodeling/deepmd-kit/tags
