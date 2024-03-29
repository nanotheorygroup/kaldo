# get_precalculated_fc.sh is used to 
# obtain precalculated force constants folder
#  for silicon bulk,using hiphive.

# Fetch precalculated force constants folder from remote
wget https://www.dropbox.com/s/8sqspll51yewsze/hiPhive_si_bulk.tar.gz?dl=0
mv hiPhive_si_bulk.tar.gz?dl=0 hiPhive_si_bulk.tar.gz

# Untar precalculated files and clean up 
tar xzvf hiPhive_si_bulk.tar.gz
rm -rf hiPhive_si_bulk.tar.gz

echo  "  "
echo "Precalculated force constant files are obtained."
