# KALDodocker

## Overview
KALDodocker is a Docker container designed for computational simulations, integrating LAMMPS with Python support and KALDo software. This container is tailored for CPU environments but also includes instructions for adapting it for GPU usage.

## Getting Started

### CPU Build

To prepare the Docker environment for CPU-based computations, rename the Dockerfile designated for CPU usage:

```bash
mv Dockerfile-cpu Dockerfile
```

#### Building the Container

Build the Docker image using the following command:

```bash
docker build -t kaldo .
```

#### Running the Container

To run KALDo in an interactive shell environment, execute:

```bash
docker run -it --rm -u $(id -u):$(id -g) kaldo /bin/bash
```

This command grants you access to a full Ubuntu environment within the container.

##### File Transfer Support

To mount your current directory inside the container for easy file transfer, use the `-v` option:

```bash
docker run -it --rm -u $(id -u):$(id -g) -v $(pwd):/root/ kaldo /bin/bash
```

This mounts your present working directory to `/root` inside the Docker container.

### GPU Build

For enabling GPU support, follow these instructions but ensure you have `nvidia-cuda-toolkit` and compatible drivers installed.

#### Preparing the Dockerfile

Switch to the Dockerfile intended for GPU use:

```bash
mv Dockerfile-gpu Dockerfile
```

#### Building the Container

To build the GPU-enabled Docker image, use:

```bash
docker build -t kaldo-gpu .
```

#### Running the Container

To run the container with GPU support:

```bash
docker run -it --rm -u $(id -u):$(id -g) --gpus all kaldo-gpu /bin/bash
```

The `--gpus all` flag enables all available GPUs. For specific GPUs, use `--gpus=0,1` to utilize GPU 0 and GPU 1, for example.

### Additional Information

- For comprehensive details on Docker and GPU support, refer to the [NVIDIA Container Toolkit documentation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) and [TensorFlow Docker installation guide](https://www.tensorflow.org/install/docker).
- The development version (`devel`) includes the entire KALDo package and is intended for debugging or when full access to KALDo is necessary.

