Mercury Artifact for SOSP'25
==============

# 1. Overview

This artifact is open sourced at `https://github.com/ChandlerGuan/mercury_artifact`.
In this artifact, we target the available and functional badges of the proposed Mercury compiler.

# 2. Environment Requirements

## 2.1. Operating System
The artifact is developed and tested on **Ubuntu 22.04**.

## 2.2. Hardware Environment
The experiments are conducted on servers equipped with:
- **CPU**: AMD EPYC 9534 64-Core Processor
- **GPU**: 8 x NVIDIA H100 80GB HBM3 interconnected with NVLink.

The code can be adapted to other multi-GPU environments, but performance may vary. A minimal setup requires at least 2 GPUs.

# 3. Installation

To evaluate the artifact, we provide a Docker image that contains the required environment and scripts.
Please run the following commands in the artifact folder to build and start the Docker container.

```
docker build -t mercury_artifact .
```

Then, start the Docker container:

```
docker run -it --rm --gpus all mercury_artifact
```

# 4. Execution

## 4.1. Common setup

In the following commands, several common command line arguments are used to specify the execution environment settings. `--nnodes` specifies the number of nodes to run the code, and `--nproc_per_node` specifies the number of processes to run on each node. The `CUDA_VISIBLE_DEVICES` environment variable is used to specify which GPUs to use for the execution.

## 4.2. End-to-end example

To run the end-to-end code generation example with the proposed CommIR, you can execute the following command inside the Docker container. Please change the `--nproc_per_node` parameters according to your hardware configuration.

```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nnodes=1 --nproc_per_node=8 ./example.py
```

This example parallelizes an attention kernel to multiple GPUs and applies a pre-defined ring-attention style communication transformation schedule.

### Expected Output
The script will first print the generated PyTorch code for the parallelized attention kernel. Then, it will execute the code. A successful run will complete without any errors and the output diff aganist the single node flash attention should be relatively small.

## 4.3. Unit tests

We provide a set of unit tests to verify the correctness of different components of the Mercury compiler.
We highlight several important unit tests that can be run as follows.

To validate the correctness of the CommIR transformation in the enumerated search space, run the validation unit test:

```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nnodes=1  --nproc_per_node=8 tests/test_search_validation.py
```

To run the actual search process, you can execute the following unit test. Note that this is a test run with a small search space to demonstrate the search process. In practice, you can use a larger search space for better performance.

```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nnodes=1  --nproc_per_node=8 tests/test_search_gemm.py
```

### Expected Output
For all unit tests, a successful execution will run a series of checks and exit without stating any errors.

# 5. Code Structure

The Mercury codebase is organized into the following main folders:

- `benchmark/`: Contains scripts for performance evaluation of different parallelization strategies and generated kernels.
- `mercury/`: The core implementation of the Mercury compiler.
  - `frontend/`: Parses high-level model descriptions (e.g., from DSLs) and lowers them into Mercury's internal representation (CommIR).
  - `ir/`: Defines the Communication-aware Intermediate Representation (CommIR), its nodes, and transformation passes for optimizing communication.
  - `backend/`: Generates target-specific code (e.g., PyTorch with `torch.distributed`) from the optimized CommIR.
  - `search/`: Implements the search algorithm to explore the space of possible parallelization strategies and find efficient communication schedules.
- `tests/`: Includes a comprehensive suite of unit tests to ensure the correctness of the IR, transformations, code generation, and search components.
- `utils/`: Provides common helper functions and Domain-Specific Language (DSL) examples for defining models like attention and GEMM.
- `example.py`: An end-to-end demo that showcases how to use Mercury to parse, transform, and generate code for a parallel attention kernel.

# 6. Minimal Working Example

The file [`example.py`](example.py) serves as a minimal working example. It demonstrates the full pipeline:
1.  Defining a model using the provided DSL.
2.  Lowering the model to CommIR.
3.  Applying a manual parallelization and communication schedule.
4.  Generating and executing the final PyTorch code.

### How to Extend
You can extend this example by:
- **Modifying the Model**: Change the parameters or structure of the `flash_attn_pack_kv_template` function within [`example.py`](example.py) to define a different model.
- **Exploring Transformations**: Modify the transformation schedule applied in the example. Instead of a manual schedule, you can integrate the search module (`mercury.search`) to automatically find an optimal schedule.
- **Defining New Kernels**: Use the DSL helpers in `utils/` to define new computational kernels and write a new script similar to `example.py` to parallelize them.

# 7. License

This project is licensed under the MIT License. Please see the [`LICENSE`](LICENSE) file for details. We welcome the community to use, compare, and extend this artifact for research purposes.

# 8. Other Q&A

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