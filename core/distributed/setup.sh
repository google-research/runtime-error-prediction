# Copyright (C) 2021 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

echo "Starting"

sudo apt-get update

sudo apt install software-properties-common -y
sudo apt install git zip emacs -y
sudo apt install make build-essential libssl-dev zlib1g-dev \
libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev -y
sudo apt install libncurses5-dev libgdbm-dev libnss3-dev libtk8.6 libgdm-dev libdb4o-cil-dev libpcap-dev -y

sudo apt install libgraphviz-dev -y

sudo apt install gcc -y

sudo apt install python3 python3-dev python3-venv -y
sudo apt install graphviz-dev -y

# Install Python 3.9
wget https://www.python.org/ftp/python/3.9.1/Python-3.9.1.tgz
tar -xf Python-3.9.1.tgz
pushd Python-3.9.1
./configure --enable-optimizations
make -j 8
sudo make altinstall
popd

# Make a virtual environment
python3.9 -m venv ipagnn
source ./ipagnn/bin/activate

pushd runtime-error-prediction
python3.9 -m pip install -r requirements.txt
python3.9 setup.py develop
popd
