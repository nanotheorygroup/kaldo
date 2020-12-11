# get_reference.sh is used to 
# obtain reference calculations
# for example silicon_clathrate_Tersoff_LAMMPS

# Fetch precalculated force constants folder from remote
wget http://sophe.ucdavis.edu/structures/reference/updated_example/silicon_clathrate_Tersoff_LAMMPS.tar.gz
# Untar precalculated files and clean up 
tar xzvf silicon_clathrate_Tersoff_LAMMPS.tar.gz
rm -rf silicon_clathrate_Tersoff_LAMMPS.tar.gz
echo  "  "
echo "Reference calculation files are obtained."
