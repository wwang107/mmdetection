ARG PYTORCH="1.3"
ARG CUDA="10.1"
ARG CUDNN="7"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

RUN apt-get update && apt-get install -y libglib2.0-0 libsm6 libxrender-dev libxext6 wget\
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# Install mmdetection
RUN conda clean --all
RUN git clone https://github.com/wwang107/mmdetection.git /mmdetection
WORKDIR /mmdetection
ENV FORCE_CUDA="1"
RUN pip install --no-cache-dir -e .
RUN pip install Cython
RUN pip install "git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI"
# Download Model Checkpoint
RUN mkdir checkpoints
RUN wget -P checkpoints/ https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/rpn_r50_fpn_1x_20181010-4a9c0712.pth

# Install visdom from source
RUN git clone https://github.com/wwang107/visdom.git
WORKDIR /mmdetection/visdom
RUN git pull
RUN pip install -e .
WORKDIR /mmdetection
# Install tmux
RUN apt-get update && apt-get install -y tmux

RUN adduser --disabled-password --gecos -u $UID user
USER user

# -------- END OF USER-DEFINED COMMANDS --------
# FOLLOWING, ONLY GENERATED COMMANDS ARE PERMITTED!


