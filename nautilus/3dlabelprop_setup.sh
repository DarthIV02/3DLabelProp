apt-get -y install libmlpack-dev
apt -y install python3-pybind11
apt-get -y install liblapack-dev
apt-get -y install libblas-dev
apt-get -y install libarmadillo-dev
# sudo apt-get install libmlpack-dev # add again if crashes with fatal error: mlpack/methods/kmeans/kmeans.hpp
#conda create --name 3DLabelProp python=3.7
#conda activate 3DLabelProp
#cd /home
#git clone https://github.com/DarthIV02/3DLabelProp.git
#cd 3DLabelProp/
echo Y | pip install -r requirements.txt
echo Y | conda install pytorch==1.7.1 torchvision torchaudio cudatoolkit=11.0 -c pytorch
cd cpp_wrappers
bash compile_wrappers.sh
pip install --force-reinstall torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117
conda install -c torchhd torchhd
export LD_LIBRARY_PATH=/root/anaconda3/envs/3DLabelProp/lib:$LD_LIBRARY_PATH


# >kubectl cp --retries 10 semantickitti.zip label-prop-5c586cbd99-9snt6:/home/
