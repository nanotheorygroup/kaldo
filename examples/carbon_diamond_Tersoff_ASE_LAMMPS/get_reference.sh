# get_reference.sh is used to 
# obtain reference calculations
#  for example 1

# Fetch precalculated force constants folder from remote
wget http://sophe.ucdavis.edu/structures/reference/reference_1_carbon_diamond_Tersoff_ASE_LAMMPS.tar.gz

# Untar precalculated files and clean up 
tar xzvf reference_1_carbon_diamond_Tersoff_ASE_LAMMPS.tar.gz
rm -rf reference_1_carbon_diamond_Tersoff_ASE_LAMMPS.tar.gz

echo  "  "
echo "Reference calculation files are obtained."
