# get_reference.sh is used to 
# obtain reference calculations
#  for example carbon_diamond_Tersoff_ASE_LAMMPS

# Fetch precalculated force constants folder from remote
wget https://www.dropbox.com/s/kvtuedw27acs8tw/carbon_diamond_Tersoff_ASE_LAMMPS.tar.gz?dl=0
mv carbon_diamond_Tersoff_ASE_LAMMPS.tar.gz?dl=0 carbon_diamond_Tersoff_ASE_LAMMPS.tar.gz

# Untar precalculated files and clean up 
tar xzvf carbon_diamond_Tersoff_ASE_LAMMPS.tar.gz
rm -rf carbon_diamond_Tersoff_ASE_LAMMPS.tar.gz

echo  "  "
echo "Reference calculation files are obtained."
