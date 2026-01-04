# TunaEvolve

ShinkaEvolve with Reinforcement Learning Fine-Tuning


### Quickstart

```bash
# `uv` will create the environment and install the dependencies
uv sync

# Run the programs using `uv run`
uv run shinka_launch [args...]
uv run shinka_visualize [args...]
```

To start the vLLM server you can use

```bash
uv run shinka/launch/vllm.py --model qwen/qwen3-0.6b --port 8000
```

where you can replace `qwen/qwen3-0.6b` with the corresponding model id.

Note that this is a simplified API, so the model id is used both to find the
model weights and as the model name for the server.

For example, to use `google/gemma-3-12b-it` as the main model for ShinkaEvolve,
you should follow the following steps:
1. Set `llm_models: ["local-google/gemma-3-12b-it"]` in the evolution config
2. Launch the vLLM server using

```bash
uv run shinka/launch/vllm.py --model google/gemma-3-12b-it --port 8000
```

3. Launch ShinkaEvolve using the given evolution config.

## Multiple local models

You may mix and match local models and remote models, but please
keep in mind that using multiple local models requires passing
an extra `--gpu-memory-utilization` argument to each vLLM server.

For example, to use `google/gemma-3-12b-it` and `qwen/qwen3-4b` as
local models, you should follow the following steps:

1. Set `llm_models: ["local-google/gemma-3-12b-it", "local-qwen/qwen3-4b"]` in
the evolution config
2. Launch two vLLM servers from two different ports, dividing the GPU memory accordingly.
Ideally, the sum of the memory utilizations shouldn't exceed `0.9`.

```bash
# start a server on port 8000 using 60% of GPU memory
uv run shinka/launch/vllm.py --model google/gemma-3-12b-it --port 8000 --gpu-memory-utilization 0.6

# in a separate terminal, start a server on port 8001 using 30% of GPU memory
uv run shinka/launch/vllm.py --model qwen/qwen3-4b --port 8001 --gpu-memory-utilization 0.3
```
