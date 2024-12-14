# Start with the Ubuntu 22.04 base image with DeepMD-kit and GPU support
FROM ghcr.io/deepmodeling/deepmd-kit:devel_cu11

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    TZ=Etc/UTC \
    LANG=en_US.utf8 \
    DEBIAN_FRONTEND=noninteractive \
    LD_LIBRARY_PATH=/usr/lib/openmpi/lib:$LD_LIBRARY_PATH \
    PATH=/usr/lib/openmpi/bin:$PATH

# Update and install system dependencies
RUN apt-get update && \
    apt-get install -y locales && \
    localedef -i en_US -c -f UTF-8 -A /usr/share/locale/locale.alias en_US.UTF-8 && \
    apt-get install -y \
        vim curl python3 python3-pip \
        build-essential gfortran cmake ccache \
        libglib2.0-0 libgl1 libegl1 libxkbcommon0 dbus libopengl0 \
        libblas-dev liblapack-dev libfftw3-dev libfftw3-mpi-dev \
        libhdf5-dev libhdf5-openmpi-dev openmpi-bin libopenmpi-dev \
        python3.11-venv curl \
        git ffmpeg && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /mnt/work

RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    /opt/deepmd-kit/bin/python3 get-pip.py && \
    rm get-pip.py

# Install Kaldo
RUN python3 -m pip install --no-cache-dir git+https://github.com/nanotheorygroup/kaldo

# Install additional Python packages for LAMMPS
RUN  python3 -m pip install calorine virtualenv psutil tabulate mpi4py Cython fitsnap3 xarray pytest numpy netcdf4 zarr

# Clone and install LAMMPS with specific packages enabled, including GPU support
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
        -DPKG_ML-SNAP=yes \
        -DPKG_ML-IAP=yes \
        -DPKG_ML-PACE=yes \
        -DPKG_SPIN=yes \
        -DPYTHON_EXECUTABLE:FILEPATH=$(which python3) && \
    make -j $(nproc) && \
    make install-python

# Clone and build TDEP (Temperature Dependent Effective Potential) package
WORKDIR /opt
RUN git clone https://github.com/tdep-developers/tdep.git
WORKDIR /opt/tdep
COPY ./important_settings /opt/tdep/.
RUN bash build_things.sh --nthreads_make $(nproc)

# Configure environment for TDEP
RUN echo 'MANPATH=$MANPATH:/opt/tdep/man' >> ~/.bashrc && \
    echo 'PATH=$PATH:/opt/tdep/bin' >> ~/.bashrc && \
    echo 'TDEP_BIN_DIR=/opt/tdep/bin' >> ~/.bashrc && \
    echo 'export MANPATH' >> ~/.bashrc && \
    echo 'export PATH' >> ~/.bashrc && \
    echo 'alias gnuplot="gnuplot -persist"' >> ~/.bashrc

# Clone Kaldo repository for running tests

# Run tests to verify tdep
WORKDIR /opt/tdep/tests
RUN make all

# Install TDEP python wrappers
RUN python3 -m pip install --no-cache-dir https://github.com/flokno/tools.tdep/archive/main.zip

# Get kaldo run tests of kaldo
WORKDIR /opt/
RUN git clone https://github.com/nanotheorygroup/kaldo
WORKDIR /opt/kaldo/.
RUN pytest


# Default command
CMD ["bash"]

