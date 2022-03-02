pip install "jax[tpu]>=0.2.16" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

# Clone repo
git clone https://github.com/google-research/runtime-error-prediction.git

# Install deps
cd runtime-error-prediction
git pull
sudo apt-get update
sudo apt install libgraphviz-dev -y
python3 -m pip install -r requirements.txt

# Connect to GCS Bucket for data
if [ ! -f /mnt/python-runtime-errors/README.md ]; then
  sudo mkdir -p /mnt/python-runtime-errors
  sudo chown $(whoami) /mnt/python-runtime-errors
  gcsfuse python-runtime-errors /mnt/python-runtime-errors/
fi

# # If writing experiment results to a bucket, mount that here as well:
# EXPERIMENT_DATA_BUCKET=runtime-error-problems-experiments
# if [ ! -f /mnt/$EXPERIMENT_DATA_BUCKET/README.md ]; then
#   sudo mkdir -p /mnt/$EXPERIMENT_DATA_BUCKET
#   sudo chown $(whoami) /mnt/$EXPERIMENT_DATA_BUCKET
#   gcsfuse $EXPERIMENT_DATA_BUCKET /mnt/$EXPERIMENT_DATA_BUCKET/
# fi

# # Uncomment to copy data out of bucket for faster access.
# if [ ! -d project-codenet-data/full-noudf-ids ]; then
#   mkdir -p project-codenet-data
#   cp -r /mnt/python-runtime-errors/datasets/project-codenet/2021-12-29 project-codenet-data/2021-12-29
# fi
