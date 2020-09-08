# get_precalculated_fc.sh is used to 
# obtain precalculated forceconstant folder
#  for amorphous silicon sample.

# Clean up existed fc folder
rm -rf fc_aSi512

# Fetch precalculated forceconstant folder from remote
wget http://sophe.ucdavis.edu/structures/fc_aSi512.tar.gz

# Untar precalculated files and clean up 
tar xzvf fc_aSi512.tar.gz
rm -rf fc_aSi512.tar.gz

echo  "  "
echo "Precalculated force constant files are obtained."
