# get_reference.sh is used to 
# obtain reference calculations
# for 10,0 carbon nanotube sample.
# Fetch precalculated force constants folder from remote
wget https://www.dropbox.com/s/2dxobxjxzsgnn8y/carbon_nanotube_Tersoff_LAMMPS_1_1_3.tar.gz?dl=0
mv carbon_nanotube_Tersoff_LAMMPS_1_1_3.tar.gz?dl=0 carbon_nanotube_Tersoff_LAMMPS_1_1_3.tar.gz
# Untar precalculated files and clean up 
tar xzvf carbon_nanotube_Tersoff_LAMMPS_1_1_3.tar.gz
mv carbon_nanotube_Tersoff_LAMMPS_1_1_3 carbon_nanotube_Tersoff_LAMMPS
rm -rf carbon_nanotube_Tersoff_LAMMPS.tar.gz
mv carbon_nanotube_Tersoff_LAMMPS/ALD_CNT_1_1_3 carbon_nanotube_Tersoff_LAMMPS/ALD_CNT
echo  "  "
echo "Reference calculation files are obtained."
