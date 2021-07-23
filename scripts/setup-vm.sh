sudo apt-get update

sudo apt install software-properties-common
sudo apt install git zip emacs
sudo apt-get install make build-essential libssl-dev zlib1g-dev \
libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev

sudo apt install gcc

sudo apt install python3 python3-dev python3-venv
sudo apt install graphviz-dev

wget https://bootstrap.pypa.io/get-pip.py
sudo python3 get-pip.py

python3 -m venv ipagnn
source ./ipagnn/bin/activate

pyenv install 3.9.6
