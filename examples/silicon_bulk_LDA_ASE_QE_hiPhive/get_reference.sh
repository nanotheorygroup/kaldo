# get_reference.sh is used to 
# obtain reference calculations
# for example silicon_bulk_LDA_QE_hiPhive

# Fetch precalculated force constants folder from remote
wget https://www.dropbox.com/s/bvxk7zkcyv8d3ak/silicon_bulk_LDA_ASE_QE_hiPhive.tar.gz?dl=0
mv silicon_bulk_LDA_ASE_QE_hiPhive.tar.gz?dl=0 silicon_bulk_LDA_ASE_QE_hiPhive.tar.gz  

# Untar precalculated files and clean up 
tar xzvf silicon_bulk_LDA_ASE_QE_hiPhive.tar.gz
rm -rf silicon_bulk_LDA_ASE_QE_hiPhive.tar.gz
echo  "  "
echo "Reference calculation files are obtained."
