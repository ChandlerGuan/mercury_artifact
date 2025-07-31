FROM nvcr.io/nvidia/pytorch:25.04-py3
WORKDIR /workspace
RUN git clone https://github.com/mercury-anonymous/mercury.git
WORKDIR /workspace/mercury
RUN pip install -e .