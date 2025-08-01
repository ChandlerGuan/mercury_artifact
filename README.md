Mercury Artifact for SOSP'25
==============

# 1. Overview

This artifact is open sourced at `https://github.com/ChandlerGuan/mercury_artifact`.
In this artifact, we target the available and functional badges of the proposed Mercury compiler.


# 2. Installation

To evaluate the artifact, we provide a Docker image that contains the required environment and scripts.
Please run the following commands in the artifact folder to build and start the Docker container.

```
docker build -t mercury_artifact .
```

Then, start the Docker container:

```
docker run -it --rm --gpus all mercury_artifact
```

# 3. Execution

## 3.1 Common setup

In the following commands, several common command line arguments are used to specify the execution environment settings. `--nnodes` specifies the number of nodes to run the code, and `--nproc_per_node` specifies the number of processes to run on each node. The `CUDA_VISIBLE_DEVICES` environment variable is used to specify which GPUs to use for the execution.

## 3.2 End-to-end example

To run the end-to-end code generation example with the proposed CommIR, you can execute the following command inside the Docker container. Please change the `--nproc_per_node` parameters according to your hardware configuration.

```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nnodes=1 --nproc_per_node=8 ./example.py
```

This example parallelize an attention kernel to multiple GPUs and applies a pre-defined ring-attention style communication transformation schedule.

## 3.3 Unit tests

We provide a set of unit tests to verify the correctness of different components of the Mercury compiler.
We hightlight several important unit tests that can be run as follows.

To validate the correctness of the CommIR transformation in the enumerated search space, run the validation unit test:

```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nnodes=1  --nproc_per_node=8 tests/test_search_validation.py
```

To run the actual search process, you can execute the following unit test. Note that this is a test run with a small search space to demonstrate the search process. In practice, you can use a larger search space for better performance.

```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nnodes=1  --nproc_per_node=8 tests/test_search_gemm.py
```

# 4. Code Structure

The Mercury codebase is organized into the following main folders:

- `benchmark/`: Scripts for performance evaluation.
- `mercury/`: Core compiler implementation.
  - `frontend/`: Parses and lowers inputs.
  - `ir/`: Intermediate representation and transformations.
  - `backend/`: Codegen and backend-specific logic.
  - `search/`: Search and scheduling for parallelization.
- `utils/`: Common helper functions and DSL examples.
- `tests/`: Unit tests for verifying correctness.
- `example.py`: An end-to-end demo using Mercury.

# Other Q&A

## Docker hub login

Since we are using a base docker image provided by Nvidia, you need to sign in to the Nvidia docker hub before building the image.
To do this:

1. Create a free NVIDIA NGC account at https://ngc.nvidia.com/signup if you don't have one.
2. Get your NGC API key from https://ngc.nvidia.com/setup/api-key.
3. Log in to the NGC registry using Docker (replace <your_api_key> with your actual key):

```
docker login nvcr.io -u '$oauthtoken' -p <your_api_key>
```

4. Re-run the docker build command:

```
docker build -t mercury_artifact .
```