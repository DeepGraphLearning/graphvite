set -e

mkdir -p build

cd build
cmake .. -DALL_ARCH=True
make
cd -

cd python
$PYTHON setup.py install
cd -