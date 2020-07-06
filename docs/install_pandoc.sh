RELEASES_URL='https://github.com/jgm/pandoc/releases'

# the 'latest' URL redirects to the name of the latest tag.
export PANDOCVERSION=$(curl -I "$RELEASES_URL/latest" | sed -ne 's#Location:.*tag/\(.*\)$#\1#p' | tr -d "\n\r")

# Show pandoc version in logs
echo $PANDOCVERSION

# downloads and extract
wget $RELEASES_URL/download/$PANDOCVERSION/pandoc-$PANDOCVERSION-linux-amd64.tar.gz
tar xvzf pandoc-$PANDOCVERSION-linux-amd64.tar.gz

# add executable to PATH
export PATH=$HOME/pandoc-$PANDOCVERSION/bin:$PATH