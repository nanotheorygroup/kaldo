### Quick Install

kALDo supports Python version 3.10 or higher. 

We recommend creating a new environment with Python 3.13.
```bash
conda create -n kaldo python=3.13
```
and enable the environment
```bash
conda activate kaldo
conda install pip
```

kALDo installation can be done using `pip`
```bash
pip install kaldo
```

#### Using `pip` and `virtualenv`

You can also install kALDo without using `conda`
```bash
pip3 install virtualenv
virtualenv --system-site-packages -p python3 ~/kaldo
source ~/kaldo/bin/activate
pip3 install git+https://github.com/nanotheorygroup/kaldo
```
#### Development mode

The best way to run examples, tests and to develop kaldo is to follow the quick install procedure, and add the following extra steps.
```bash
pip uninstall kaldo
mkdir ~/develoment
cd ~/development
git clone https://github.com/nanotheorygroup/kaldo
export PYTHONPATH=~/development/kaldo:$PYTHONPATH
```
If you followed the steps in the quickstart and then uninstall kaldo, you will have all the dependencies correctly installed.
The next lines are pulling the repo from Github and adding it to the `PYTHONPATH`.

If you want to make the last change in the `PYTHONPATH` permanent, you can also run
```bash
echo "export PYTHONPATH=~/development/kaldo:$PYTHONPATH" >> ~/.bashrc
```
