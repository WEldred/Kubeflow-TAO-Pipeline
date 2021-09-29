FROM nvcr.io/nvidia/tao/tao-toolkit-tf:v3.21.08-py3

# Update repository
RUN apt update

# Install Tao software
RUN pip3 install nvidia-pyindex
RUN pip3 install nvidia-tao
RUN pip3 install kfp

# Setup mountpoint for model and data files
RUN mkdir /mnt/workspace

# Install ngc
#
# Note: Standard container installs NGC CLI at docker run, but Kubeflow does not run entry scripts,
#       so we are installing it ahead of time.

WORKDIR /opt/
RUN mkdir ngccli
WORKDIR /opt/ngccli
RUN wget https://ngc.nvidia.com/downloads/ngccli_reg_linux.zip
RUN unzip ngccli_reg_linux.zip

# Go back to workspace
WORKDIR /workspace/

