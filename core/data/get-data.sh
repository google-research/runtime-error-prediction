# Run from compressive-ipagnn root dir.
# This will download the data from Dropbox and unzip it into the data directory
# After running, check that data/description2code_current is populated.
if [ ! -f description2code_current.zip ]; then
  curl -L -O https://www.dropbox.com/s/zwj6u4caehf54s0/description2code_current.zip
fi
num_files=($(unzip -l description2code_current.zip | wc -l))  # 616353 files in zip.
unzip description2code_current.zip -d data | pv -l -s ${num_files} > /dev/null
