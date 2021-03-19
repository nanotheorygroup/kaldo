# get_precalculated_fc.sh is used to 
# obtain precalculated force constants folder
#  for type 1 clathrate sample.

# Clean up existed fc folder
rm -rf fc_Si46

# Fetch precalculated force constants folder from remote
wget https://www.dropbox.com/s/r80yvlsmyzb7bqo/fc_Si46.tar.gz?dl=0
mv fc_Si46.tar.gz?dl=0 fc_Si46.tar.gz

# Untar precalculated files and clean up 
tar xzvf fc_Si46.tar.gz
rm -rf fc_Si46.tar.gz

echo  "  "
echo "Precalculated force constant files are obtained."
