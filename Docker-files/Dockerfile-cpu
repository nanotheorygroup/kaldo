# Use Ubuntu 22.04 as the base image
FROM ubuntu:22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    TZ=Etc/UTC \
    LANG=en_US.utf8

# Update and install system dependencies
RUN apt-get update && \
    apt-get install -y locales && \
    localedef -i en_US -c -f UTF-8 -A /usr/share/locale/locale.alias en_US.UTF-8 && \
    apt-get install -y vim curl python3 python3-pip \
        libglib2.0-0 libgl1 libegl1 libxkbcommon0 dbus libopengl0 build-essential git \
        ccache openmpi-bin libopenmpi-dev cmake libblas-dev liblapack-dev python3.10-venv && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory to /mnt/work
WORKDIR /mnt/work

# Copy the Python requirements file
COPY requirements.txt /mnt/work/

# Install Python packages without caching
RUN python3 -m pip install --no-cache-dir git+https://github.com/nanotheorygroup/kaldo && \
    python3 -m pip install --no-cache-dir -r requirements.txt

# Clone, build, and install LAMMPS
RUN git clone https://github.com/lammps/lammps.git lammps && \
    cd lammps && \
    mkdir build && \
    cd build && \
    cmake ../cmake \
        -DBUILD_SHARED_LIBS=yes \
        -DMLIAP_ENABLE_PYTHON=yes \
        -DPKG_PYTHON=yes \
        -DPKG_MANYBODY=yes \
        -DPKG_KSPACE=yes \
        -DPKG_PHONON=yes \
        -DPYTHON_EXECUTABLE:FILEPATH=$(which python3) && \
    make -j $(nproc) && \
    make install-python

# Set the working directory back to the root
WORKDIR /root/

