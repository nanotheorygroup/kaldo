# get_precalculated_fc.sh is used to 
# obtain precalculated force constants folder
#  for amorphous silicon sample.

# Clean up existed fc folder
rm -rf fc_aSi512

# Fetch precalculated force constants folder from remote
wget https://www.dropbox.com/s/9ecdycd33bcjarh/fc_aSi512.tar.gz?dl=0
mv fc_aSi512.tar.gz?dl=0 fc_aSi512.tar.gz

# Untar precalculated files and clean up 
tar xzvf fc_aSi512.tar.gz
rm -rf fc_aSi512.tar.gz

echo  "  "
echo "Precalculated force constant files are obtained."
