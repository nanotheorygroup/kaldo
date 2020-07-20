rm -rf autosummary
make clean
make html
cp -a _build/html/* .