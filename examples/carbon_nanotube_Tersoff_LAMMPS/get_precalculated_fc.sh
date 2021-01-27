# get_precalculated_fc.sh is used to 
# obtain precalculated force constants folder
# for 10,0 carbon nanotube sample.
# Clean up existed fc folder
rm -rf fc_CNT
# Fetch precalculated force constants folder from remote
wget http://sophe.ucdavis.edu/structures/fc_CNT_1_1_3.tar.gz

# Untar precalculated files and clean up 
tar xzvf fc_CNT_1_1_3.tar.gz
rm -rf fc_CNT_1_1_3.tar.gz

echo  "  "
echo "Precalculated force constant files are obtained."
