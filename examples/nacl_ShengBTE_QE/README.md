# Source environment variables to help setup input/output folders
# This keeps the data organized
# If you want to see which variables we set please open the file 
# as text. They all begin with "kaldo_" to keep them seperate from
# any variables used by other programs
source tools/vars.sh

# Unpack FC's
First, decompress the force constant information from the tarball in the
tools directory. ("tools/nacl-sheng-qe.tgz" ~ 1 Mb)
mkdir ${kaldo_inputs}
tar -xvzf tools/nacl-sheng-qe.tgz -C ${kaldo_iputs}


# Run kALDo simulations!
# With NAC
python 1_run_kaldo.py nac
# No NAC
python 1_run_kaldo.py


