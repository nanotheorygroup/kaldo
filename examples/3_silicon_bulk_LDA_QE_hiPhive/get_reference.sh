# get_reference.sh is used to 
# obtain reference calculations
# for example 3

# Fetch precalculated forceconstant folder from remote
wget http://sophe.ucdavis.edu/structures/reference/reference_3_silicon_bulk_LDA_QE_hiPhive.tar.gz
# Untar precalculated files and clean up 
tar xzvf reference_3_silicon_bulk_LDA_QE_hiPhive.tar.gz
rm -rf reference_3_silicon_bulk_LDA_QE_hiPhive.tar.gz
echo  "  "
echo "Reference calculation files are obtained."
