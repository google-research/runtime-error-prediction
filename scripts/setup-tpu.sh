echo "Starting"
hostname
pip install "jax[tpu]>=0.2.16" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

# Clone repo
git clone https://$PERSONAL_ACCESS_TOKEN@github.com/googleprivate/compressive-ipagnn.git

# Install deps
cd compressive-ipagnn
git fetch --all
git checkout 2021-11-09-binary-sweep
git pull
sudo apt-get update
sudo apt install libgraphviz-dev tidy -y
python3 -m pip install -r requirements.txt

# Connect to GCS Bucket
# fusermount -u /mnt/runtime-error-problems-experiments
if [ ! -f /mnt/runtime-error-problems-experiments/README.md ]; then
  sudo mkdir -p /mnt/runtime-error-problems-experiments
  sudo chown $(whoami) /mnt/runtime-error-problems-experiments
  gcsfuse --implicit-dirs runtime-error-problems-experiments /mnt/runtime-error-problems-experiments/
fi

# # Copy data out of bucket for faster access.
# if [ ! -d project-codenet-data/full-noudf-ids ]; then
#   mkdir -p project-codenet-data
#   cp -r /mnt/runtime-error-problems-experiments/datasets/project-codenet/full-noudf-ids project-codenet-data/full-noudf-ids
# fi
hostname
echo "Done"
