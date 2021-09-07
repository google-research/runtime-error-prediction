pip install "jax[tpu]>=0.2.16" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

# Clone repo
git clone https://$PERSONAL_ACCESS_TOKEN@github.com/googleprivate/compressive-ipagnn.git

# Install deps
cd compressive-ipagnn
sudo apt-get update
sudo apt install libgraphviz-dev -y
python3 -m pip install -r requirements.txt

# Connect to GCS Bucket
sudo mkdir -p /mnt/runtime-error-problems-experiments
sudo chown $(whoami) /mnt/runtime-error-problems-experiments
gcsfuse runtime-error-problems-experiments /mnt/runtime-error-problems-experiments/

# TODO(dbieber): Possibly copy data out of bucket for faster access.
