# get_precalculated_fc.sh is used to 
# obtain precalculated force constants folder
# for 10,0 carbon nanotube sample.
# Clean up existed fc folder
rm -rf fc_CNT
# Fetch precalculated force constants folder from remote
wget https://www.dropbox.com/s/3as8n3uu9zdxpmg/fc_CNT_1_1_3.tar.gz?dl=0
mv fc_CNT_1_1_3.tar.gz?dl=0 fc_CNT_1_1_3.tar.gz

# Untar precalculated files and clean up 
tar xzvf fc_CNT_1_1_3.tar.gz
mv fc_CNT_1_1_3 fc_CNT
rm -rf fc_CNT_1_1_3.tar.gz

echo  "  "
echo "Precalculated force constant files are obtained."
