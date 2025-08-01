FROM nvcr.io/nvidia/pytorch:25.04-py3
WORKDIR /workspace
RUN git clone https://github.com/ChandlerGuan/mercury_artifact.git
WORKDIR /workspace/mercury
RUN pip install -e .