echo -y | apt-get install libmlpack-dev
apt -y install python3-pybind11
apt-get install liblapack-dev
apt-get install libblas-dev
apt-get install libarmadillo-dev
conda create --name 3DLabelProp python=3.7
conda activate 3DLabelProp
cd /home
git clone https://github.com/DarthIV02/3DLabelProp.git
cd 3DLabelProp/
pip install -r requirements.txt
echo -y | conda install pytorch==1.7.1 torchvision torchaudio cudatoolkit=11.0 -c pytorch
pip install torch-sparse
pip install pybind11
cd cpp_wrappers
bash compile_wrappers.sh

# >kubectl cp --retries 10 semantickitti.zip label-prop-5c586cbd99-9snt6:/home/
