# EvoFlowSearch

Code for the paper **"EvoFlowSearch: Execution-Grounded Workflow Search for Compact Tool-Using Agents"**.

We frame agentic tool-use as search over a graph of typed tool calls. A planner LLM generates a workflow graph for each query, and an evolutionary loop mutates + scores candidates to find better ones. Evaluation runs on [MCP-Bench](https://github.com/accenture/mcp-bench).

---

## Setup

```bash
pip install -r requirements.txt
pip install vllm          # only needed for local inference
```

Put your keys in a `.env` file at the repo root:

```
OPENROUTER_API_KEY=...
HF_TOKEN=...
GOOGLE_MAPS_API_KEY=...    # needed by some MCP-Bench servers
NASA_API_KEY=...
# AWS / Azure keys if using those providers
```

---

## Running

### Quickest test (no external servers needed)

```bash
PYTHONPATH=. python3 -m scripts.run_pipeline --benchmark toy --stages search,validate
```

This runs the full search + validation loop on the toy benchmark to make sure everything works.

---

### MCP-Bench

You'll need to clone `mcp-bench` and install its servers first. Then:

```bash
# search pass
PYTHONPATH=. python3 -m scripts.run_pipeline \
    --benchmark mcpbench \
    --mcpbench-root /path/to/mcp-bench \
    --complexity single \
    --stages search

# validation pass
PYTHONPATH=. python3 -m scripts.run_pipeline \
    --benchmark mcpbench \
    --mcpbench-root /path/to/mcp-bench \
    --complexity single \
    --stages validate
```

`--complexity` can be `single`, `2server`, `3server`, or `all`.

---

### Using `run.sh` (handles vLLM server lifecycle)

`run.sh` spins up two vLLM servers (planner on `:8000`, judge on `:8001`) and then runs the pipeline. The `STAGES` and `MODEL_VERSION` env vars control what gets run.

```bash
# local vLLM
STAGES=search   bash ./run.sh local "Qwen/Qwen2.5-7B-Instruct" "openai/gpt-4o-mini"
STAGES=validate bash ./run.sh local "Qwen/Qwen2.5-7B-Instruct" "openai/gpt-4o-mini"

# OpenRouter
bash ./run.sh openrouter

# AWS Bedrock
bash ./run.sh bedrock
```

For fine-tuned checkpoints, set `MODEL_VERSION`:

```bash
MODEL_VERSION=sft     STAGES=validate bash ./run.sh local "Qwen/Qwen2.5-7B-Instruct" "openai/gpt-4o-mini"
MODEL_VERSION=sft+dpo STAGES=validate bash ./run.sh local "Qwen/Qwen2.5-7B-Instruct" "openai/gpt-4o-mini"
```

Results land in `results/{MODEL_VERSION}/`. Logs go to `log/`.

---

### ReAct baseline

To run the ReAct loop instead of graph search:

```bash
PYTHONPATH=. python3 -m scripts.run_pipeline --benchmark mcpbench \
    --mcpbench-root /path/to/mcp-bench \
    --stages react
```

---

### Fine-tuning on search trajectories

After running search:

```bash
PYTHONPATH=. python3 -m scripts.build_training_dataset
PYTHONPATH=. python3 -m scripts.train_sft_dpo
```

Training config is in [configs/fsdp_config.yaml](configs/fsdp_config.yaml).

---

## Key flags

**Search:**
- `--B 15` — evolution budget (iterations per query)
- `--K 15` — max population size
- `--max-steps 25` — max tool calls per execution
- `--seed 0`

**ReAct:**
- `--react-max-steps 25`
- `--react-max-tokens 3096`
