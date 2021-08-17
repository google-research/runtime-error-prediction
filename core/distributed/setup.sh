echo "Starting"
apt install libgraphviz-dev
pushd compressive-ipagnn
python setup.py develop
popd
