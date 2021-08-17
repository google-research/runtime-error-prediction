echo "Starting"

sudo apt-get update

sudo apt install software-properties-common -y
sudo apt install git zip emacs -y
sudo apt install make build-essential libssl-dev zlib1g-dev \
libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev -y
sudo apt install libncurses5-dev libgdbm-dev libnss3-dev -y


sudo apt install libgraphviz-dev -y

sudo apt install gcc -y

sudo apt install python3 python3-dev python3-venv -y
sudo apt install graphviz-dev -y

# wget https://bootstrap.pypa.io/get-pip.py
wget https://www.python.org/ftp/python/3.9.1/Python-3.9.1.tgz
tar -xf Python-3.9.1.tgz
cd Python-3.9.1
./configure --enable-optimizations
make -j 8
sudo make altinstall


# pushd compressive-ipagnn
# python setup.py develop
# popd
