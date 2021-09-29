FROM nvcr.io/nvidia/tao/tao-toolkit-tf:v3.21.08-py3

# Update repository
RUN apt update

# Install Tao software
RUN pip3 install nvidia-pyindex
RUN pip3 install nvidia-tao

# Create directories
RUN mkdir /workspace/notebooks
RUN mkdir /workspace/notebooks/specs
COPY classification.ipynb /workspace/notebooks
COPY specs/classification_spec.cfg /workspace/notebooks/specs
COPY specs/classification_retrain_spec.cfg /workspace/notebooks/specs

# Install ngc
WORKDIR /opt/
RUN mkdir ngccli
WORKDIR /opt/ngccli
RUN wget https://ngc.nvidia.com/downloads/ngccli_reg_linux.zip
RUN unzip ngccli_reg_linux.zip

# Go back to workspace
WORKDIR /workspace/

# Notes from Kubeflow custom container page
ENV NB_PREFIX /
CMD ["bash","-c", "source activate rapids && jupyter lab --notebook-dir=/workspace/notebooks --ip=0.0.0.0 --no-browser --allow-root --port=8888 --LabApp.token='' --LabApp.password='' --LabApp.allow_origin='*' --LabApp.base_url=${NB_PREFIX}"]

