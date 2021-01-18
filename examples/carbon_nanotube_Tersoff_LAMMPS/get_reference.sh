# get_reference.sh is used to 
# obtain reference calculations
# for 10,0 carbon nanotube sample.
# Fetch precalculated force constants folder from remote
wget http://sophe.ucdavis.edu/structures/reference/updated_example/carbon_nanotube_Tersoff_LAMMPS.tar.gz
# Untar precalculated files and clean up 
tar xzvf carbon_nanotube_Tersoff_LAMMPS.tar.gz
rm -rf carbon_nanotube_Tersoff_LAMMPS.tar.gz
echo  "  "
echo "Reference calculation files are obtained."