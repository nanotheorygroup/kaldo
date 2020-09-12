# get_precalculated_fc.sh is used to 
# obtain precalculated force constants folder
#  for silicon bulk,using hiphive.

# Fetch precalculated force constants folder from remote
wget http://sophe.ucdavis.edu/structures/hiphive_si_bulk.tar.gz
# Untar precalculated files and clean up 
tar xzvf hiphive_si_bulk.tar.gz
rm -rf hiphive_si_bulk.tar.gz

echo  "  "
echo "Precalculated force constant files are obtained."
