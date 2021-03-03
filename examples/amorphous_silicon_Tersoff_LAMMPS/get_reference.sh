# get_reference.sh is used to 
# obtain reference calculations
#  for example amorphous_silicon_Tersoff_LAMMPS


# Fetch precalculated forceconstant folder from remote
wget https://www.dropbox.com/s/8lvs173z8tjlw0m/amorphous_silicon_Tersoff_LAMMPS.tar.gz?dl=0
mv amorphous_silicon_Tersoff_LAMMPS.tar.gz?dl=0 amorphous_silicon_Tersoff_LAMMPS.tar.gz
# Untar precalculated files and clean up 
tar xzvf amorphous_silicon_Tersoff_LAMMPS.tar.gz
rm -rf amorphous_silicon_Tersoff_LAMMPS.tar.gz
echo  "  "
echo "Reference calculation files are obtained."
