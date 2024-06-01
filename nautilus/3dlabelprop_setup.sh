echo -y | apt-get install libmlpack-dev
apt -y install python3-pybind11
apt-get install liblapack-dev
apt-get install libblas-dev
apt-get install libarmadillo-dev
conda create --name 3DLabelProp python=3.7
conda activate 3DLabelProp
cd /home
