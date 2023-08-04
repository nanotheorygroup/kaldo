### Readme.md

Seemed funny to use a bash script to use a bash script but this is the order of commands I ran to start from a fresh pull
on the amd-ryzen machine, and then run it completely.
The final output shows a bunch of force mismatches because the tolerances are 0 right now.
Good luck! slack me if you need!!

# 0
# this part is pretty un-breakable
cp modded_kaldo/harmonic_with_q.py ../../kaldo/observables/harmonic_with_q.py
cp modded_espresso/matdyn.f90 <pathtoqe>/PHonon/PH/
pushd !$
cd ../../
make ph

# 1
# but can't find a good way to find your path super easily.
# either copy the output from the echo or replace the path
# in the sed command. make sure you escape directory slashes as seen below
# a
echo $( realpath bin/matdyn.x )
popd
#    use vi to edit the REPLACEME string   #
vi kactus.sh

# b
mdpathvariable=$( realpath bin/matdyn.x )
popd
#   not sure but you can probably use the path as a variable some how. not super sure
sed -i "s/REPLACEME/\/home\/nwlundgren\/develop\/quantumespresso\/build-6.23.23\/bin\/matdyn.x/" kactus.sh

# 2
## Okay run
./kactus.sh

