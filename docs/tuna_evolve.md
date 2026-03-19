# TunaEvolve

ShinkaEvolve with Reinforcement Learning Fine-Tuning.

## Quickstart

Before setting up this project, you should
[install `uv`](https://docs.astral.sh/uv/getting-started/installation/)
if you haven't installed it yet.

```bash
# Clone this repository in the directory of your choice
git clone https://github.com/alejandrocalles/TunaEvolve.git
cd TunaEvolve

# `uv` will create the environment and install the dependencies
uv sync

# Run the main program using `uv run`
uv run main.py

# You can also run the ShinkaEvolve scripts using `uv run`
uv run shinka_launch [args...]
uv run shinka_visualize [args...]
```

## Configuration

**Note**: this section is intended to be complementary to the ShinkaEvolve
documentation. If something seems confusing, consider referring to the
[README](../README.md) and the
[configuration guide](./configuration.md). Since this project uses
Hydra for structuring configuration files, you might find the
[Hydra documentation](https://hydra.cc/docs/intro/) useful as well.

TunaEvolve adds two core configuration components: inference and training.

### 1. Training Config

Controls the parameters for the training step of the EvoTune algorithm.

```yaml
training_config:
  _target_: shinka.train.configuration.DPOTrainingConfig
  model_id: "google/gemma-3-12b-it"                        # the model id, should match the one in the evolution config, but without any "local-" prefix
  base_output_dir: "dpo_checkpoints"                       # the name of the directory where the checkpoints should be saved, relative to the results directory
  logging_steps: 10                                        # the number of training steps between logging

  evotune:                                                 # parameters specific to the EvoTune algorithm
    _target_: shinka.train.configuration.EvoTuneConfig
    training_enabled: true                                 # whether training is enabled
    num_generations_per_period: 2                          # the number of program generations between each training period

  hyperparameters:                                         # parameters specific to DPO training
    _target_: shinka.train.configuration.DPOHyperparameters
    beta: 0.1
    learning_rate: 5.0e-6
    batch_size: 4
    gradient_accumulation_steps: 2
    epochs: 3
    lr_scheduler: "cosine"
    warmup_ratio: 0.1
    max_length: 2048
    max_prompt_length: 1024

  hardware:                                                # parameters related to hardware/precision
    _target_: shinka.train.configuration.HardwareConfig
    fp16: true                                             # whether fp16 should be used
    bf16: false                                            # whether bf16 should be used

  lora:                                                    # parameters related to Low-Rank Adaptation
    _target_: shinka.train.configuration.LoRAConfig
    enabled: true
    r: 16
    alpha: 32
    dropout: 0.05
    target_modules:
      - "q_proj"
      - "v_proj"
      - "k_proj"
      - "o_proj"
```

- For an explanation on each parameter in the `hyperparameters` entry, see the
[`DPOConfig` documentation](https://huggingface.co/docs/trl/v0.27.1/en/dpo_trainer#trl.DPOConfig)
from HuggingFace's `trl` package.

- For an explanation on each parameter in the `lora` entry, see the
[`LoraConfig` documentation](https://huggingface.co/docs/peft/en/package_reference/lora)
from HuggingFace's `peft` package.

### 2. Inference Config

Controls the parameters for launching the trained model locally (for inference) using vLLM.

```yaml
vllm_config:
  _target_: shinka.vllm.VLLMConfig
  gpu_memory_utilization: 0.9                           # the percentage of GPU memory that should be used by the KV cache
  tensor_parallel_size: 1                               # if greater than 1, enables tensor parallelism
  pipeline_parallel_size: 1                             # if greater than 1, enables pipeline parallelism
  trust_remote_code: false                              # whether to trust custom LLM architectures (usually not necessary)
  dtype: auto                                           # one of [auto, half, float16, bfloat16, float, float32]
  log_dir: vllm_logs                                    # the directory where vllm logs should be saved
```

- For tensor parallelism, if running a single model, it is recommended that the size matches the number of GPUs in the node.
If you have 4 GPUs available, set `tensor_parallel_size: 4`.
- For pipeline parallelism, if running a single model, it is recommended that the size matches the number of nodes.
If you have 2 nodes with 4 GPUs each, set `tensor_parallel_size: 4` and `pipeline_parallel_size: 2`.

For more information on parallelism, refer to the following guides:
- [vLLM Parallelism and Scaling Guide](https://docs.vllm.ai/en/stable/serving/parallelism_scaling/)
- [vLLM Distributed Serving Guide](https://docs.vllm.ai/en/v0.8.0/serving/distributed_serving.html)

