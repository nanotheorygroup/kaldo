# get_reference.sh is used to 
# obtain reference calculations
# for example 4

# Fetch precalculated forceconstant folder from remote
wget http://sophe.ucdavis.edu/structures/reference/reference_4_silicon_clathrate_Tersoff_LAMMPS.tar.gz
# Untar precalculated files and clean up 
tar xzvf reference_4_silicon_clathrate_Tersoff_LAMMPS.tar.gz
rm -rf reference_4_silicon_clathrate_Tersoff_LAMMPS.tar.gz
echo  "  "
echo "Reference calculation files are obtained."
