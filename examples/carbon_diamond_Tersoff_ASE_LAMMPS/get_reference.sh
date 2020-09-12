# get_reference.sh is used to 
# obtain reference calculations
#  for example carbon_diamond_Tersoff_ASE_LAMMPS

# Fetch precalculated force constants folder from remote
wget http://sophe.ucdavis.edu/structures/reference/updated_example/carbon_diamond_Tersoff_ASE_LAMMPS.tar.gz

# Untar precalculated files and clean up 
tar xzvf carbon_diamond_Tersoff_ASE_LAMMPS.tar.gz
rm -rf carbon_diamond_Tersoff_ASE_LAMMPS.tar.gz

echo  "  "
echo "Reference calculation files are obtained."
