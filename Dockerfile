# 必要な場合はproxy設定を追加する
# nvidia-driver >= 580 && pytorch-2.0 binary-build
ARG CUDA="12.1.0"  
ARG CUDNN="8"
ARG UBUNTU="22.04"
FROM nvidia/cuda:${CUDA}-cudnn${CUDNN}-devel-ubuntu${UBUNTU}
# rootで実行
USER root
RUN apt-get update -y && apt-get upgrade -y && apt-get autoremove -y
RUN apt-get install -y --no-install-recommends build-essential python3-dev emacs python3.9 python3-pip libsndfile1-dev libmysqlclient-dev
RUN apt-get install -y ffmpeg libopus-dev
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
RUN pip3 install PyYAML torchmetrics lightning einops numpy tqdm pandas scipy numba h5py seaborn librosa mysqlclient 'setuptools<81'
RUN pip3 install tensorboard
RUN pip3 install torchmetrics[audio] ffmpeg
RUN pip3 install transformers
RUN pip3 install peft
