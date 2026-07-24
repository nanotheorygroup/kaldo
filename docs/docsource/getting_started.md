### Quick Install

kALDo supports Python version 3.10 or higher.

Installations and package dependencies are handled by [uv](https://docs.astral.sh/uv/), please refer to the manual when needed.

We recommend creating a new environment with Python 3.13.
```bash
uv venv ~/kaldo --python 3.13
```
and enable the environment
```bash
source ~/kaldo/bin/activate
```

kALDo installation can be done using `uv`
```bash
uv pip install kaldo
```

#### Installing directly from GitHub

You can also install the latest version of kALDo directly from the repository
```bash
uv venv ~/kaldo --python 3.13
source ~/kaldo/bin/activate
uv pip install git+https://github.com/nanotheorygroup/kaldo
```

#### Development mode

The best way to run examples, tests and to develop kaldo is to follow the quick install procedure, and add the following extra steps.
```bash
uv pip uninstall kaldo
mkdir ~/development
cd ~/development
git clone https://github.com/nanotheorygroup/kaldo
uv pip install -e ~/development/kaldo
```
If you followed the steps in the quickstart and then uninstall kaldo, you will have all the dependencies correctly installed.
The next lines are pulling the repo from Github and installing it in editable mode, so any changes you make to the source code take effect immediately without needing to modify the `PYTHONPATH`.

### Examples Repository

A comprehensive examples repository is available with detailed workflows for various materials:

**Repository**: [https://github.com/nanotheorygroup/kaldo-examples](https://github.com/nanotheorygroup/kaldo-examples)

Examples include:
- Si, Ge, SiC, GaAs, MgO, AlN thermal conductivity
- DFT with DFPT/D3Q workflows
- Empirical potentials via LAMMPS
- Machine-learned potentials (MatterSim, ORB, NEP, ACE)

### Docker

A Docker image with all dependencies pre-installed is available:
```bash
docker pull gbarbalinardo/kaldo:latest
```
