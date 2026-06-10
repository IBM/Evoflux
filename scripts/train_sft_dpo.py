"""
scripts/train_sft_dpo.py
────────────────────────
Two-stage fine-tuning: SFT → DPO

Stage 1 — SFT on train_sft.jsonl
    Teaches workflow vocabulary, graph structure, and node-parameter patterns
    from sonnet's best candidates per query.

    Validation after SFT:
      • JSON syntax rate         (target ≥ 80 %)
      • Schema validity rate     (nodes/edges/required fields present)
      • Node coverage (Jaccard)  vs sonnet's reference fingerprint
      • Score proxy              ≈ node_coverage × reference_score
      • Score delta vs sonnet    (target ≤ 2.0 pts)
      • Grounding proxy          (params contain query keywords)
      • Tool Appropriateness proxy (servers match query server list)

Stage 2 — DPO on train_dpo.jsonl  (reference = frozen SFT checkpoint)
    Steers the model toward sonnet's preferred workflows via hard negatives
    selected with action_id bucket diversity (mirrors _prune()).

    Validation after DPO:
      • Reward margin P(chosen > rejected) on held-out DPO pairs
      • Re-run same structural metrics on same SFT val prompts
      • Delta vs SFT checkpoint for Grounding and Tool Appropriateness

Supported models
────────────────
  meta-llama/Llama-3.2-3B-Instruct
  Qwen/Qwen3.5-4B
  ibm-granite/granite-4.0-h-micro
  microsoft/Phi-4-mini-instruct
  HuggingFaceTB/SmolLM3-3B
  deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
  nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16

Usage
─────
  python scripts/train_sft_dpo.py \\
      --model meta-llama/Llama-3.2-3B-Instruct \\
      --sft-data   datasets/sonnet_workflow_v1/train_sft.jsonl \\
      --dpo-data   datasets/sonnet_workflow_v1/train_dpo.jsonl \\
      --ranking-data datasets/sonnet_workflow_v1/train_ranking.jsonl \\
      --output-dir /data/Kushal/AgenticWorkflow/trained/Llama-3.2-3B-Instruct

  # Resume from existing SFT checkpoint (skip SFT, run DPO only)
  python scripts/train_sft_dpo.py --model ... --skip-sft \\
      --sft-ckpt /data/Kushal/AgenticWorkflow/trained/.../sft_checkpoint

  # Inspect data + model without training
  python scripts/train_sft_dpo.py --model ... --dry-run

  # Evaluate saved checkpoints without training
  python scripts/train_sft_dpo.py --model ... --eval-only \\
      --output-dir /data/Kushal/AgenticWorkflow/trained/...
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import gc
import torch
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ── Supported models ──────────────────────────────────────────────────────────
DPO_PROMPT_TOKENS = 2000
DPO_COMPLETION_TOKENS = 500
DPO_MAX_LENGTH = DPO_PROMPT_TOKENS + DPO_COMPLETION_TOKENS

SUPPORTED_MODELS: List[str] = [
    "meta-llama/Llama-3.2-3B-Instruct",
    "Qwen/Qwen3.5-4B",
    "ibm-granite/granite-4.0-h-micro",
    "ibm-granite/granite-4.0-micro",
    "microsoft/Phi-4-mini-instruct",
    "HuggingFaceTB/SmolLM3-3B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16",
]

# Models that need trust_remote_code=True
_TRUST_REMOTE: set = {
    "ibm-granite/granite-4.0-h-micro",
    "ibm-granite/granite-4.0-micro",
    "microsoft/Phi-4-mini-instruct",
    "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16",
}

# Models that emit <think>...</think> reasoning blocks (strip before JSON parse)
_THINKING_MODELS: set = {
    "Qwen/Qwen3.5-4B",
    "ibm-granite/granite-4.0-h-micro",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
}

# LoRA targets covering qkvo + MLP projections across all target architectures
_LORA_TARGETS: List[str] = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# Workflow schema (steps-array format)
_REQUIRED_STEP_FIELDS: set = {"node_id", "server", "tool"}


# ── Dependency check ──────────────────────────────────────────────────────────

def _check_deps() -> None:
    missing = []
    for pkg in ("torch", "transformers", "trl", "peft", "datasets"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"[error] Missing packages: {', '.join(missing)}")
        print("  pip install " + " ".join(missing))
        if "torch" in missing:
            print("  CUDA: pip install torch --index-url https://download.pytorch.org/whl/cu121")
        sys.exit(1)


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_sft_data(
    path: str,
    val_frac: float = 0.12,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict]]:
    """Split train_sft.jsonl into train / val, shuffled by seed."""
    import random
    rng = random.Random(seed)
    records = _load_jsonl(path)
    rng.shuffle(records)
    n_val = max(1, int(len(records) * val_frac))
    return records[n_val:], records[:n_val]


def load_dpo_data(
    path: str,
    val_frac: float = 0.12,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict]]:
    """Split train_dpo.jsonl by query_id so full query groups stay together."""
    import random
    rng = random.Random(seed)
    records = _load_jsonl(path)
    by_query: Dict[str, List[Dict]] = defaultdict(list)
    for r in records:
        by_query[r["query_id"]].append(r)
    qids = list(by_query.keys())
    rng.shuffle(qids)
    n_val_q = max(1, int(len(qids) * val_frac))
    val_qids   = set(qids[:n_val_q])
    train_qids = set(qids[n_val_q:])
    train = [r for q in train_qids for r in by_query[q]]
    val   = [r for q in val_qids   for r in by_query[q]]
    return train, val


def load_ranking_data(path: Optional[str]) -> Dict[str, Dict[str, Any]]:
    """Load train_ranking.jsonl → {query_id: record} for subscore access."""
    if not path or not Path(path).exists():
        return {}
    return {r["query_id"]: r for r in _load_jsonl(path)}


# ── HuggingFace Dataset builders ──────────────────────────────────────────────

def build_sft_hf_dataset(records: List[Dict[str, Any]], tokenizer):
    """
    Pre-apply the chat template so SFTTrainer receives `text` (full conversation)
    and `prompt_text` (system + user only, with the generation-prompt header).
    The latter is used by CompletionOnlyCollator to mask prompt tokens to -100
    so loss is computed only on the assistant's JSON response.
    """
    from datasets import Dataset

    rows = []
    for r in records:
        messages = r["messages"]
        try:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            prompt_text = tokenizer.apply_chat_template(
                messages[:2], tokenize=False, add_generation_prompt=True
            )
        except Exception:
            text = "\n".join(
                f"{m['role'].upper()}: {m['content']}" for m in messages
            )
            prompt_text = "\n".join(
                f"{m['role'].upper()}: {m['content']}" for m in messages[:2]
            ) + "\nASSISTANT:"
        rows.append({
            "text":           text,
            "prompt_text":    prompt_text,
            "query_id":       r["query_id"],
            "positive_score": r.get("positive_score", 0.0),
        })
    return Dataset.from_list(rows)


def build_dpo_hf_dataset(records: List[Dict[str, Any]], tokenizer=None):
    """
    Build a DPO dataset with pre-formatted strings to avoid the tokenizer
    prefix-mismatch warning in DPOTrainer.
    """
    from datasets import Dataset

    rows = []
    for r in records:
        if tokenizer is not None:
            try:
                prompt_str = tokenizer.apply_chat_template(
                    r["prompt"], tokenize=False, add_generation_prompt=True
                )
                full_chosen = tokenizer.apply_chat_template(
                    r["prompt"] + r["chosen"], tokenize=False, add_generation_prompt=False
                )
                full_rejected = tokenizer.apply_chat_template(
                    r["prompt"] + r["rejected"], tokenize=False, add_generation_prompt=False
                )
                chosen_completion   = full_chosen[len(prompt_str):]
                rejected_completion = full_rejected[len(prompt_str):]
                rows.append({
                    "prompt":         prompt_str,
                    "chosen":         chosen_completion,
                    "rejected":       rejected_completion,
                    "query_id":       r["query_id"],
                    "chosen_score":   r.get("chosen_score", 0.0),
                    "rejected_score": r.get("rejected_score", 0.0),
                    "score_gap":      r.get("score_gap", 0.0),
                })
                continue
            except Exception:
                pass
        rows.append({
            "prompt":         r["prompt"],
            "chosen":         r["chosen"],
            "rejected":       r["rejected"],
            "query_id":       r["query_id"],
            "chosen_score":   r.get("chosen_score", 0.0),
            "rejected_score": r.get("rejected_score", 0.0),
            "score_gap":      r.get("score_gap", 0.0),
        })
    return Dataset.from_list(rows)

def _is_chat_messages(x):
    return (
        isinstance(x, list)
        and all(isinstance(m, dict) and "role" in m and "content" in m for m in x)
    )


def _as_assistant_text(x):
    if isinstance(x, str):
        return x

    if _is_chat_messages(x):
        # DPO chosen/rejected is usually [{"role": "assistant", "content": "..."}]
        return "\n".join(str(m.get("content", "")) for m in x).strip()

    raise TypeError(f"Unsupported chosen/rejected format: {type(x)}")


def _format_prompt(prompt, tokenizer):
    if isinstance(prompt, str):
        return prompt

    if _is_chat_messages(prompt):
        return tokenizer.apply_chat_template(
            prompt,
            tokenize=False,
            add_generation_prompt=True,
        )

    raise TypeError(f"Unsupported prompt format: {type(prompt)}")


def truncate_text_by_tokens(text, tokenizer, max_tokens, keep="end"):
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]

    if len(ids) <= max_tokens:
        return text

    if keep == "start":
        ids = ids[:max_tokens]
    elif keep == "end":
        ids = ids[-max_tokens:]
    else:
        raise ValueError("keep must be 'start' or 'end'")

    return tokenizer.decode(ids, skip_special_tokens=False)


def build_dpo_hf_dataset_capped(
    records,
    tokenizer,
    max_seq_len: int = 4608,
    min_samples: int = 0,
):
    """
    Prefer samples whose total token length (prompt + max(chosen, rejected))
    already fits within max_seq_len — no truncation needed, no information lost.

    Only if the number of fitting samples is below min_samples do we fall back
    to truncating the oversized samples (sorted by score_gap descending so the
    most informative pairs are added first).
    """
    from datasets import Dataset

    # Split budget used only when truncation is actually needed.
    completion_tokens = max(256, int(max_seq_len * 0.40))
    prompt_tokens     = max_seq_len - completion_tokens

    fitting  = []   # (row_dict,)  — samples that fit as-is
    oversized = []  # (r, prompt_str, chosen_str, rejected_str) — need truncation

    for r in records:
        prompt_str          = _format_prompt(r["prompt"], tokenizer)
        chosen_completion   = _as_assistant_text(r["chosen"])
        rejected_completion = _as_assistant_text(r["rejected"])

        prompt_len   = len(tokenizer.encode(prompt_str,          add_special_tokens=False))
        chosen_len   = len(tokenizer.encode(chosen_completion,   add_special_tokens=False))
        rejected_len = len(tokenizer.encode(rejected_completion, add_special_tokens=False))
        total_len    = prompt_len + max(chosen_len, rejected_len)

        row = {
            "prompt":         prompt_str,
            "chosen":         chosen_completion,
            "rejected":       rejected_completion,
            "query_id":       r.get("query_id", ""),
            "chosen_score":   r.get("chosen_score", 0.0),
            "rejected_score": r.get("rejected_score", 0.0),
            "score_gap":      r.get("score_gap", 0.0),
        }

        if total_len <= max_seq_len:
            fitting.append(row)
        else:
            oversized.append((r, prompt_str, chosen_completion, rejected_completion))

    print(
        f"[DPO dataset] {len(fitting)}/{len(records)} samples fit within "
        f"{max_seq_len} tokens without truncation."
    )

    rows = list(fitting)

    needed = max(0, min_samples - len(fitting))
    if needed > 0:
        # Sort by score_gap descending: add the most contrastive pairs first.
        oversized.sort(key=lambda x: x[0].get("score_gap", 0.0), reverse=True)
        to_add = oversized[:needed]
        print(
            f"  Only {len(fitting)} fitting samples < min_samples={min_samples}; "
            f"adding {len(to_add)} truncated sample(s) to compensate."
        )
        for r, prompt_str, chosen_completion, rejected_completion in to_add:
            # keep="start": preserve task + tool catalog; drop verbose trailing content.
            prompt_str = truncate_text_by_tokens(
                prompt_str, tokenizer, prompt_tokens, keep="start"
            )
            chosen_completion = truncate_text_by_tokens(
                chosen_completion, tokenizer, completion_tokens, keep="start"
            )
            rejected_completion = truncate_text_by_tokens(
                rejected_completion, tokenizer, completion_tokens, keep="start"
            )
            rows.append({
                "prompt":         prompt_str,
                "chosen":         chosen_completion,
                "rejected":       rejected_completion,
                "query_id":       r.get("query_id", ""),
                "chosen_score":   r.get("chosen_score", 0.0),
                "rejected_score": r.get("rejected_score", 0.0),
                "score_gap":      r.get("score_gap", 0.0),
            })

    return Dataset.from_list(rows)

# ── Model loading ─────────────────────────────────────────────────────────────

def _get_device_map(use_fsdp: bool) -> Optional[Any]:
    """Return the device_map to pass to from_pretrained."""
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if use_fsdp:
        if local_rank != -1:
            torch.cuda.set_device(local_rank)
        return None  # FSDP manages sharding
    if local_rank != -1:
        torch.cuda.set_device(local_rank)
        return {"": local_rank}
    # Single-GPU or CPU-only
    if torch.cuda.is_available():
        return "auto"
    return None


def load_base_model(
    model_name: str,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
    use_fsdp: bool = False,
):
    """
    Load base model + tokenizer and wrap with a fresh LoRA adapter.
    Returns (model, tokenizer).
    """
    from transformers import (
        AutoTokenizer, AutoModelForCausalLM, AutoConfig,
        BitsAndBytesConfig, GenerationConfig,
    )
    from peft import LoraConfig, get_peft_model

    trust = model_name in _TRUST_REMOTE



    # ── Sync model config + generation config with tokenizer special tokens ────
    model_config = AutoConfig.from_pretrained(model_name, trust_remote_code=trust)
    try:
        gen_config = GenerationConfig.from_pretrained(model_name, trust_remote_code=trust)
    except Exception:
        gen_config = GenerationConfig()

    # ── Tokenizer — align TO the model config, not the other way around ────────
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        # Use whatever the model config already designates as pad; fall back to eos
        if model_config.pad_token_id is not None:
            tokenizer.pad_token_id = model_config.pad_token_id
        else:
            tokenizer.pad_token    = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
            # Only here do we need to write back, because the config had nothing
            model_config.pad_token_id = tokenizer.pad_token_id
            gen_config.pad_token_id   = tokenizer.pad_token_id

    for cfg in (model_config, gen_config):
        if tokenizer.pad_token_id is not None:
            cfg.pad_token_id = tokenizer.pad_token_id
        if tokenizer.eos_token_id is not None:
            cfg.eos_token_id = tokenizer.eos_token_id
        if tokenizer.bos_token_id is not None:
            cfg.bos_token_id = tokenizer.bos_token_id

    # ── Quantization ───────────────────────────────────────────────────────────
    quant_cfg = None
    if use_fsdp and (load_in_4bit or load_in_8bit):
        print("  [warn] BitsAndBytes quantization is incompatible with FSDP — ignoring.")
    elif load_in_4bit:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    elif load_in_8bit:
        quant_cfg = BitsAndBytesConfig(load_in_8bit=True)

    device_map = _get_device_map(use_fsdp)

    model_kwargs: Dict[str, Any] = {
        "config":              model_config,
        "generation_config":   gen_config,
        "trust_remote_code":   trust,
        "torch_dtype":         torch.bfloat16,
        "attn_implementation": "eager",
        "low_cpu_mem_usage":   True,
    }
    if device_map is not None:
        model_kwargs["device_map"] = device_map
    if quant_cfg is not None:
        model_kwargs["quantization_config"] = quant_cfg
        model_kwargs.pop("torch_dtype", None)

    base = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    base.config.use_cache = False

    lora_cfg = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=_LORA_TARGETS,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base, lora_cfg)
    _cast_to_bf16(model)
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    return model, tokenizer


def load_model_from_checkpoint(
    model_name: str,
    checkpoint_path: str,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
    use_fsdp: bool = False,
):
    """
    Load base model and attach a saved LoRA adapter from checkpoint_path.
    This is used to reload the SFT checkpoint for DPO training and evaluation.
    Returns (model, tokenizer).
    """
    from transformers import (
        AutoTokenizer, AutoModelForCausalLM, AutoConfig,
        BitsAndBytesConfig, GenerationConfig,
    )
    from peft import PeftModel

    trust = model_name in _TRUST_REMOTE

    

    model_config = AutoConfig.from_pretrained(model_name, trust_remote_code=trust)
    try:
        gen_config = GenerationConfig.from_pretrained(model_name, trust_remote_code=trust)
    except Exception:
        gen_config = GenerationConfig()
    
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_path,  # load tokenizer from checkpoint so vocab is consistent
        trust_remote_code=trust,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token    = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    for cfg in (model_config, gen_config):
        if tokenizer.pad_token_id is not None:
            cfg.pad_token_id = tokenizer.pad_token_id
        if tokenizer.eos_token_id is not None:
            cfg.eos_token_id = tokenizer.eos_token_id
        if tokenizer.bos_token_id is not None:
            cfg.bos_token_id = tokenizer.bos_token_id

    quant_cfg = None
    if use_fsdp and (load_in_4bit or load_in_8bit):
        print("  [warn] BitsAndBytes quantization is incompatible with FSDP — ignoring.")
    elif load_in_4bit:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    elif load_in_8bit:
        quant_cfg = BitsAndBytesConfig(load_in_8bit=True)

    device_map = _get_device_map(use_fsdp)

    model_kwargs: Dict[str, Any] = {
        "config":              model_config,
        "generation_config":   gen_config,
        "trust_remote_code":   trust,
        "torch_dtype":         torch.bfloat16,
        "attn_implementation": "eager",
        "low_cpu_mem_usage":   True,
    }
    if device_map is not None:
        model_kwargs["device_map"] = device_map
    if quant_cfg is not None:
        model_kwargs["quantization_config"] = quant_cfg
        model_kwargs.pop("torch_dtype", None)

    base = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    base.config.use_cache = False

    # Load the saved LoRA adapter weights from checkpoint
    model = PeftModel.from_pretrained(
        base,
        checkpoint_path,
        is_trainable=True,   # keep adapter trainable so DPO can update it
    )
    _cast_to_bf16(model)
    model.enable_input_require_grads()

    # Confirm the model is on GPU if available
    device = next(model.parameters()).device
    print(f"  [load] Model loaded from {checkpoint_path} → device={device}")

    return model, tokenizer


# ── FSDP layer-class detection ─────────────────────────────────────────────────

_MODEL_TYPE_TO_LAYER_CLS: dict = {
    "qwen3":      "Qwen3_5DecoderLayer",
    "qwen3_5":    "Qwen3_5DecoderLayer",
    "qwen2":      "Qwen2DecoderLayer",
    "qwen2_moe":  "Qwen2MoeDecoderLayer",
    "granite":    "GraniteMoeHybridDecoderLayer",
    "granitemoe": "GraniteMoeHybridDecoderLayer",
    "phi3":       "Phi3DecoderLayer",
    "phi4":       "Phi3DecoderLayer",
    "llama":      "LlamaDecoderLayer",
    "mistral":    "MistralDecoderLayer",
    "nemotron":   "NemotronDecoderLayer",
    "deepseekr1-Qwen": "Qwen2DecoderLayer",
    "smolm3": "SmolLM3DecoderLayer"
}


def _cast_to_bf16(model) -> None:
    """
    Cast every floating-point parameter and buffer to bfloat16 in-place.

    model.to(torch.bfloat16) calls each module's .to() which can be
    intercepted by PEFT hooks and leave some LoRA matrices in float32.
    Writing directly to .data bypasses those hooks and guarantees a uniform
    dtype across all tensors — required by FSDP before it flattens them.
    """
    for param in model.parameters():
        if param.is_floating_point():
            param.data = param.data.to(torch.bfloat16)
    for buf in model.buffers():
        if buf.is_floating_point():
            buf.data = buf.data.to(torch.bfloat16)


def _get_fsdp_layer_cls(model) -> str:
    model_type = getattr(getattr(model, "config", None), "model_type", "") or ""
    cls_name = _MODEL_TYPE_TO_LAYER_CLS.get(model_type.lower())
    if cls_name:
        return cls_name
    from collections import Counter
    counts: Counter = Counter()
    for _, mod in model.named_modules():
        name = type(mod).__name__
        if any(kw in name.lower() for kw in ("layer", "block", "decoder")):
            counts[name] += 1
    if counts:
        return counts.most_common(1)[0][0]
    raise ValueError(
        f"Cannot determine FSDP transformer_layer_cls_to_wrap for model_type='{model_type}'. "
        "Add it to _MODEL_TYPE_TO_LAYER_CLS."
    )


# ── Completion-only data collator ─────────────────────────────────────────────

class CompletionOnlyCollator:
    """
    Tokenizes full `text` (system + user + assistant) but sets labels to -100
    for prompt tokens so loss is computed only on the assistant's JSON response.

    When the full sequence exceeds max_length, tokens are removed from the LEFT
    of the prompt (not the right of the response) so the response always survives.
    """

    def __init__(self, tokenizer, max_length: int) -> None:
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        input_ids_list: List[torch.Tensor] = []
        labels_list:    List[torch.Tensor] = []

        for feature in features:
            full_ids   = self.tokenizer(feature["text"],        return_tensors="pt")["input_ids"][0]
            prompt_ids = self.tokenizer(feature["prompt_text"], return_tensors="pt")["input_ids"][0]

            n_prompt     = len(prompt_ids)
            response_ids = full_ids[n_prompt:]

            if len(full_ids) > self.max_length:
                available_for_prompt = self.max_length - len(response_ids)
                if available_for_prompt <= 0:
                    response_ids     = response_ids[:self.max_length]
                    prompt_ids_trunc = full_ids[:0]
                else:
                    prompt_ids_trunc = prompt_ids[-available_for_prompt:]
                combined       = torch.cat([prompt_ids_trunc, response_ids])
                n_prompt_final = len(prompt_ids_trunc)
            else:
                combined       = full_ids
                n_prompt_final = n_prompt

            lbl = combined.clone()
            lbl[:n_prompt_final] = -100

            input_ids_list.append(combined)
            labels_list.append(lbl)

        pad_id = self.tokenizer.pad_token_id or 0
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids_list, batch_first=True, padding_value=pad_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels_list, batch_first=True, padding_value=-100
        )
        attention_mask = (input_ids != pad_id).long()
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


# ── Training stats callback ───────────────────────────────────────────────────

from transformers import TrainerCallback as _TrainerCallback


class StatsCallback(_TrainerCallback):
    """Captures every logged training/eval event into Python lists.

    Inherits from TrainerCallback so the Trainer accepts it and all
    unimplemented hook methods fall through to TrainerCallback's no-ops.
    """

    def __init__(self) -> None:
        self.step_logs:     List[Dict[str, Any]] = []
        self.epoch_logs:    List[Dict[str, Any]] = []
        self.train_summary: Dict[str, Any]       = {}
        self._start_time: Optional[float]        = None

    def on_train_begin(self, _args, _state, _control, **_kwargs):
        import time
        self._start_time = time.time()

    def on_log(self, _args, state, _control, logs=None, **_kwargs):
        import time
        if logs is None:
            return
        entry = {"step": state.global_step, "epoch": state.epoch}
        entry.update({k: v for k, v in logs.items() if isinstance(v, (int, float, bool))})
        if self._start_time is not None:
            entry["elapsed_s"] = round(time.time() - self._start_time, 1)
        if any(k.startswith("eval_") for k in logs):
            self.epoch_logs.append(entry)
        else:
            self.step_logs.append(entry)

    def on_train_end(self, _args, state, _control, **_kwargs):
        self.train_summary = {
            "total_steps":     state.global_step,
            "best_metric":     state.best_metric,
            "best_model_ckpt": state.best_model_checkpoint,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_logs":     self.step_logs,
            "epoch_logs":    self.epoch_logs,
            "train_summary": self.train_summary,
        }


def save_training_stats(stats: Dict[str, Any], ckpt_dir: str, name: str = "training_stats.json") -> None:
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    step_losses  = [e["loss"]      for e in stats.get("step_logs",  []) if "loss"      in e]
    eval_losses  = [e["eval_loss"] for e in stats.get("epoch_logs", []) if "eval_loss" in e]
    total_steps  = stats.get("train_summary", {}).get("total_steps", 0)
    elapsed_list = [e.get("elapsed_s", 0) for e in stats.get("step_logs", []) if "elapsed_s" in e]

    curve: Dict[str, Any] = {}
    if step_losses:
        curve["train_loss_start"] = round(step_losses[0],  4)
        curve["train_loss_end"]   = round(step_losses[-1], 4)
        curve["train_loss_min"]   = round(min(step_losses), 4)
    if eval_losses:
        curve["eval_loss_start"]  = round(eval_losses[0],  4)
        curve["eval_loss_end"]    = round(eval_losses[-1], 4)
        curve["eval_loss_min"]    = round(min(eval_losses), 4)
    if elapsed_list and total_steps:
        total_elapsed = elapsed_list[-1]
        curve["total_elapsed_s"]  = round(total_elapsed, 1)
        curve["steps_per_second"] = round(total_steps / max(total_elapsed, 1), 2)

    payload = {**stats, "curve_summary": curve}
    path = os.path.join(ckpt_dir, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"  [stats] Training stats → {path}")


def print_curve_summary(stats: Dict[str, Any], label: str) -> None:
    curve   = stats.get("curve_summary", {})
    summary = stats.get("train_summary", {})
    print(f"\n  [{label.upper()}] Training curve")
    if "train_loss_start" in curve:
        print(f"    train loss  {curve['train_loss_start']:.4f} → "
              f"{curve['train_loss_end']:.4f}  (min {curve['train_loss_min']:.4f})")
    if "eval_loss_start" in curve:
        print(f"    eval  loss  {curve['eval_loss_start']:.4f} → "
              f"{curve['eval_loss_end']:.4f}  (min {curve['eval_loss_min']:.4f})")
    if "total_elapsed_s" in curve:
        mins = curve["total_elapsed_s"] / 60
        print(f"    elapsed     {curve['total_elapsed_s']:.0f}s  ({mins:.1f} min)  "
              f"{curve.get('steps_per_second', 0):.2f} steps/s")
    if summary.get("best_metric") is not None:
        print(f"    best metric {summary['best_metric']:.4f}  "
              f"@ {summary.get('best_model_ckpt', 'N/A')}")


# ── Structural validation helpers ─────────────────────────────────────────────

def _strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _parse_workflow(text: str) -> Optional[Dict[str, Any]]:
    text = _strip_thinking(text)
    if "```" in text:
        lines = text.split("\n")
        text = "\n".join(l for l in lines if not l.startswith("```")).strip()
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
    return None


def _check_schema(wf: Dict[str, Any]) -> Tuple[bool, List[str]]:
    v: List[str] = []
    if "steps" not in wf:
        v.append("missing 'steps'")
    steps = wf.get("steps", [])
    if not isinstance(steps, list) or not steps:
        v.append("steps must be a non-empty list")
    else:
        for i, s in enumerate(steps):
            if not isinstance(s, dict):
                v.append(f"step[{i}] is not a dict")
                continue
            if "node_id" not in s:
                v.append(f"step[{i}] missing 'node_id'")
            if s.get("kind") != "output":
                missing = _REQUIRED_STEP_FIELDS - set(s.keys())
                if missing:
                    v.append(f"step[{i}] missing {missing}")
            if "requires" in s and not isinstance(s["requires"], list):
                v.append(f"step[{i}] 'requires' must be a list")
    return len(v) == 0, v


def _node_fingerprint(wf: Optional[Dict[str, Any]]) -> frozenset:
    if not wf:
        return frozenset()
    return frozenset(
        (s.get("server", ""), s.get("tool", ""))
        for s in wf.get("steps", [])
        if s.get("kind") != "output"
    )


def _jaccard(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 1.0
    union = len(a | b)
    return len(a & b) / union if union > 0 else 0.0


def _extract_servers_from_user_msg(user_msg: str) -> List[str]:
    if "Available tool servers:" not in user_msg:
        return []
    line = [l for l in user_msg.split("\n") if "Available tool servers:" in l]
    if not line:
        return []
    part = line[0].split("Available tool servers:")[-1].strip()
    return [s.strip() for s in part.split(",") if s.strip()]


def _grounding_proxy(wf: Dict[str, Any], query_text: str) -> float:
    stopwords = {"the", "a", "an", "of", "for", "in", "to", "and", "or",
                 "is", "be", "on", "at", "by", "as", "with", "it", "its"}
    query_words = {w.lower() for w in query_text.split() if w.lower() not in stopwords}
    tool_steps = [s for s in wf.get("steps", []) if s.get("kind") != "output"]
    if not tool_steps:
        return 0.0
    grounded = 0
    for s in tool_steps:
        params = s.get("params") or {}
        param_text = " ".join(str(v) for v in params.values()).lower()
        if query_words & set(param_text.split()):
            grounded += 1
    return grounded / len(tool_steps)


def _tool_appropriateness_proxy(wf: Dict[str, Any], available_servers: List[str]) -> float:
    tool_steps = [s for s in wf.get("steps", []) if s.get("kind") != "output"]
    if not tool_steps:
        return 0.0
    avail = {srv.lower() for srv in available_servers}
    appropriate = sum(1 for s in tool_steps if s.get("server", "").lower() in avail)
    return appropriate / len(tool_steps)


# ── Generation ────────────────────────────────────────────────────────────────

def generate_completions(
    model,
    tokenizer,
    prompt_message_lists: List[List[Dict[str, str]]],
    max_new_tokens: int = 1500,
    batch_size: int = 2,
) -> List[str]:
    """Generate assistant completions for a list of [system, user] message lists."""
    model.eval()
    device = next(model.parameters()).device
    completions: List[str] = []

    for i in range(0, len(prompt_message_lists), batch_size):
        batch = prompt_message_lists[i: i + batch_size]
        texts: List[str] = []
        for msgs in batch:
            try:
                text = tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                text = (
                    "\n".join(f"{m['role'].upper()}: {m['content']}" for m in msgs)
                    + "\nASSISTANT:"
                )
            texts.append(text)

        enc = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2500,
            truncation_side="left",
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        input_len = enc["input_ids"].shape[1]
        for seq in out:
            decoded = tokenizer.decode(seq[input_len:], skip_special_tokens=True)
            completions.append(decoded.strip())

        torch.cuda.empty_cache()

    model.train()
    return completions


def compute_log_prob(
    model,
    tokenizer,
    prompt_msgs: List[Dict[str, str]],
    completion: str,
) -> float:
    """Mean per-token log probability of `completion` given `prompt_msgs`."""
    import torch.nn.functional as F

    try:
        full_text = tokenizer.apply_chat_template(
            prompt_msgs + [{"role": "assistant", "content": completion}],
            tokenize=False, add_generation_prompt=False,
        )
        prompt_text = tokenizer.apply_chat_template(
            prompt_msgs, tokenize=False, add_generation_prompt=True,
        )
    except Exception:
        full_text = (
            "\n".join(f"{m['role'].upper()}: {m['content']}" for m in prompt_msgs)
            + f"\nASSISTANT: {completion}"
        )
        prompt_text = (
            "\n".join(f"{m['role'].upper()}: {m['content']}" for m in prompt_msgs)
            + "\nASSISTANT:"
        )

    device = next(model.parameters()).device
    full_enc   = tokenizer(full_text,   return_tensors="pt", truncation=True, max_length=2048)
    prompt_enc = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=2048)

    input_ids = full_enc["input_ids"].to(device)
    n_prompt  = prompt_enc["input_ids"].shape[1]

    with torch.no_grad():
        logits = model(input_ids=input_ids).logits

    shift_logits = logits[0, n_prompt - 1: -1, :]
    shift_labels = input_ids[0, n_prompt:]

    if len(shift_labels) == 0:
        return 0.0

    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_lp  = log_probs[torch.arange(len(shift_labels)), shift_labels]
    return token_lp.mean().item()


# ── Validation ────────────────────────────────────────────────────────────────

def validate_checkpoint(
    model,
    tokenizer,
    sft_val_records: List[Dict[str, Any]],
    ranking_data: Dict[str, Any],
    label: str,
    max_gen: int = 20,
    gen_batch: int = 2,
) -> Dict[str, Any]:
    """
    Generate completions on the SFT val set and compute structural metrics.
    Works for both the SFT and DPO checkpoints.
    """
    subset = sft_val_records[:max_gen]
    prompts = [r["messages"][:2] for r in subset]
    print(f"  [{label}] Generating {len(subset)} validation completions "
          f"(device={next(model.parameters()).device})...")
    completions = generate_completions(model, tokenizer, prompts, batch_size=gen_batch)

    n_syntax = n_schema = 0
    cov_sum = proxy_sum = delta_sum = ground_sum = approp_sum = 0.0
    per_example: List[Dict[str, Any]] = []

    for rec, completion in zip(subset, completions):
        qid       = rec["query_id"]
        ref_score = rec.get("positive_score", 0.0)

        rank_rec = ranking_data.get(qid, {})
        if rank_rec:
            ref_wf = rank_rec.get("positive", {}).get("workflow")
        else:
            ref_text = rec["messages"][2]["content"] if len(rec["messages"]) > 2 else ""
            ref_wf   = _parse_workflow(ref_text)

        gen_wf     = _parse_workflow(completion)
        valid_json = gen_wf is not None

        valid_schema, schema_viols = (False, ["no workflow parsed"])
        if gen_wf:
            valid_schema, schema_viols = _check_schema(gen_wf)

        ref_fp   = _node_fingerprint(ref_wf)
        gen_fp   = _node_fingerprint(gen_wf)
        coverage = _jaccard(ref_fp, gen_fp)
        proxy    = coverage * ref_score
        delta    = ref_score - proxy

        user_msg      = rec["messages"][1]["content"] if len(rec["messages"]) > 1 else ""
        avail_servers = _extract_servers_from_user_msg(user_msg)
        grounding     = _grounding_proxy(gen_wf, user_msg)               if gen_wf else 0.0
        approp        = _tool_appropriateness_proxy(gen_wf, avail_servers) if gen_wf else 0.0

        if valid_json:   n_syntax  += 1
        if valid_schema: n_schema  += 1
        cov_sum    += coverage
        proxy_sum  += proxy
        delta_sum  += delta
        ground_sum += grounding
        approp_sum += approp

        print("QID: ",qid)
        print("ref_score",ref_score)
        print("Generated Text",completion)
        print("Generated Workflow",gen_wf)
        print("+++"*100,flush=True)
        per_example.append({
            "query_id":          qid,
            "ref_score":         ref_score,
            "valid_json":        valid_json,
            "valid_schema":      valid_schema,
            "schema_violations": schema_viols,
            "node_coverage":     round(coverage, 4),
            "score_proxy":       round(proxy, 4),
            "score_delta":       round(delta, 4),
            "grounding_proxy":   round(grounding, 4),
            "tool_approp_proxy": round(approp, 4),
        })

    n = len(subset)
    return {
        "label":             label,
        "n_examples":        n,
        "syntax_rate":       round(n_syntax  / n, 4),
        "schema_rate":       round(n_schema  / n, 4),
        "node_coverage":     round(cov_sum   / n, 4),
        "score_proxy":       round(proxy_sum / n, 4),
        "score_delta":       round(delta_sum / n, 4),
        "grounding_proxy":   round(ground_sum / n, 4),
        "tool_approp_proxy": round(approp_sum / n, 4),
        "per_example":       per_example,
    }


def validate_reward_margin(
    model,
    tokenizer,
    dpo_val_records: List[Dict[str, Any]],
    max_pairs: int = 40,
) -> Dict[str, Any]:
    """Compute reward margins on held-out DPO preference pairs."""
    model.eval()
    subset = dpo_val_records[:max_pairs]
    print(f"  [dpo] Computing reward margins on {len(subset)} preference pairs "
          f"(device={next(model.parameters()).device})...")

    margins:          List[float] = []
    grounding_deltas: List[float] = []
    approp_deltas:    List[float] = []
    n_correct = 0

    for rec in subset:
        user_msg      = rec["prompt"][1]["content"] if len(rec["prompt"]) > 1 else ""
        avail_servers = _extract_servers_from_user_msg(user_msg)

        chosen_text = (
            rec["chosen"] if isinstance(rec["chosen"], str)
            else (rec["chosen"][0]["content"] if rec["chosen"] else "")
        )
        rejected_text = (
            rec["rejected"] if isinstance(rec["rejected"], str)
            else (rec["rejected"][0]["content"] if rec["rejected"] else "")
        )

        chosen_wf   = _parse_workflow(chosen_text)
        rejected_wf = _parse_workflow(rejected_text)

        try:
            lp_c = compute_log_prob(model, tokenizer, rec["prompt"], chosen_text)
            lp_r = compute_log_prob(model, tokenizer, rec["prompt"], rejected_text)
            margin = lp_c - lp_r
            margins.append(margin)
            if margin > 0:
                n_correct += 1
        except Exception as exc:
            print(f"    [warn] log_prob failed: {exc}")

        g_c = _grounding_proxy(chosen_wf,   user_msg)               if chosen_wf   else 0.0
        g_r = _grounding_proxy(rejected_wf,  user_msg)               if rejected_wf  else 0.0
        a_c = _tool_appropriateness_proxy(chosen_wf,   avail_servers) if chosen_wf   else 0.0
        a_r = _tool_appropriateness_proxy(rejected_wf,  avail_servers) if rejected_wf  else 0.0
        grounding_deltas.append(g_c - g_r)
        approp_deltas.append(a_c - a_r)

    n  = len(subset)
    nm = len(margins)
    model.train()

    return {
        "n_pairs":              n,
        "reward_margin_mean":   round(sum(margins) / nm, 4)           if nm else None,
        "reward_margin_pos":    round(n_correct / nm, 4)              if nm else None,
        "grounding_delta_mean": round(sum(grounding_deltas) / n, 4)   if n  else None,
        "approp_delta_mean":    round(sum(approp_deltas) / n, 4)      if n  else None,
    }


# ── Reporting ─────────────────────────────────────────────────────────────────

def _bar(v: float, width: int = 25) -> str:
    return "█" * int(v * width) + "░" * (width - int(v * width))


def print_checkpoint_report(metrics: Dict[str, Any]) -> None:
    label = metrics["label"].upper()
    print(f"\n{'─'*62}")
    print(f"  {label} Validation  ({metrics['n_examples']} examples)")
    print(f"{'─'*62}")
    print(f"  JSON syntax rate      {metrics['syntax_rate']:.1%}  {_bar(metrics['syntax_rate'])}")
    print(f"  Schema validity       {metrics['schema_rate']:.1%}  {_bar(metrics['schema_rate'])}")
    print(f"  Node coverage (Jacc)  {metrics['node_coverage']:.3f}  {_bar(metrics['node_coverage'])}")
    print(f"  Score proxy (mean)    {metrics['score_proxy']:.2f}  (ref avg × coverage)")
    print(f"  Score Δ vs sonnet     {metrics['score_delta']:.2f}  (target ≤ 2.0)")
    print(f"  Grounding proxy       {metrics['grounding_proxy']:.3f}  {_bar(metrics['grounding_proxy'])}")
    print(f"  Tool Approp proxy     {metrics['tool_approp_proxy']:.3f}  {_bar(metrics['tool_approp_proxy'])}")
    ok_syntax = metrics["syntax_rate"] >= 0.80
    ok_score  = metrics["score_delta"] <= 2.0
    print(f"\n  {'✓' if ok_syntax else '✗'} Syntax ≥ 80 %:    {metrics['syntax_rate']:.1%}")
    print(f"  {'✓' if ok_score  else '✗'} Score Δ ≤ 2.0:   {metrics['score_delta']:.2f}")
    print(f"{'─'*62}")


def print_dpo_comparison(
    sft_m:    Dict[str, Any],
    dpo_m:    Dict[str, Any],
    margin_m: Dict[str, Any],
) -> None:
    print(f"\n{'─'*62}")
    print(f"  DPO Validation  ({margin_m['n_pairs']} preference pairs + checkpoint diff)")
    print(f"{'─'*62}")
    rm = margin_m.get("reward_margin_mean")
    rp = margin_m.get("reward_margin_pos")
    print(f"  Reward margin (mean)    {rm:+.4f}" if rm is not None else "  Reward margin (mean)    N/A")
    print(f"  Chosen ranked higher    {rp:.1%}"   if rp is not None else "  Chosen ranked higher    N/A")
    gd = margin_m.get("grounding_delta_mean")
    ad = margin_m.get("approp_delta_mean")
    print(f"  Grounding Δ(ch-rej)     {gd:+.3f}" if gd is not None else "  Grounding Δ(ch-rej)     N/A")
    print(f"  Tool Approp Δ(ch-rej)   {ad:+.3f}" if ad is not None else "  Tool Approp Δ(ch-rej)   N/A")

    print(f"\n  {'Metric':<26} {'SFT':>8}  {'DPO':>8}  {'Δ':>8}")
    print(f"  {'─'*52}")
    for key, label in [
        ("grounding_proxy",   "Grounding proxy"),
        ("tool_approp_proxy", "Tool Approp proxy"),
        ("score_proxy",       "Score proxy"),
        ("syntax_rate",       "Syntax rate"),
        ("node_coverage",     "Node coverage"),
    ]:
        sv    = sft_m.get(key, 0.0)
        dv    = dpo_m.get(key, 0.0)
        delta = dv - sv
        sign  = "+" if delta >= 0 else ""
        print(f"  {label:<26} {sv:>8.3f}  {dv:>8.3f}  {sign}{delta:.3f}")

    ok_margin    = (rm or 0) > 0
    ok_grounding = (dpo_m.get("grounding_proxy",  0) >= sft_m.get("grounding_proxy",   0))
    ok_approp    = (dpo_m.get("tool_approp_proxy", 0) >= sft_m.get("tool_approp_proxy", 0))
    print(f"\n  {'✓' if ok_margin    else '✗'} Positive reward margin")
    print(f"  {'✓' if ok_grounding  else '✗'} Grounding improved vs SFT")
    print(f"  {'✓' if ok_approp     else '✗'} Tool Approp improved vs SFT")
    print(f"{'─'*62}")


def save_report(report: Dict[str, Any], out_dir: str, name: str = "training_report.json") -> None:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    path = os.path.join(out_dir, name)
    slim = json.loads(json.dumps(report))
    for stage_key in ("final_validation", "sft_validation", "dpo_validation"):
        if isinstance(slim.get(stage_key), dict):
            slim[stage_key].pop("per_example", None)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(slim, f, indent=2, ensure_ascii=False)
    print(f"  [save] Report → {path}")


def _save_eval_details(metrics: Dict[str, Any], ckpt_dir: str, name: str) -> None:
    per_example = metrics.get("per_example", [])
    if not per_example:
        return
    path = os.path.join(ckpt_dir, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(per_example, f, indent=2, ensure_ascii=False)
    print(f"  [eval] Per-example details → {path}")


# ── SFT training ──────────────────────────────────────────────────────────────

def run_sft(
    model,
    tokenizer,
    train_ds,
    val_ds,
    output_dir: str,
    epochs: int        = 3,
    batch_size: int    = 2,
    grad_accum: int    = 8,
    lr: float          = 2e-4,
    logging_steps: int = 10,
    max_seq_len: int   = 7000,
    use_fsdp: bool     = False,
    # Needed only when use_fsdp=True: used to create a fresh model after training
    # so that FSDP wrappers are stripped before DPO.
    model_name: str    = "",
    lora_r: int        = 16,
    lora_alpha: int    = 32,
    lora_dropout: float = 0.05,
) -> Tuple[str, Dict[str, Any], Any]:
    """
    SFT fine-tune. Returns (checkpoint_path, stats_dict, clean_model).
    """
    from trl import SFTTrainer, SFTConfig

    sft_out = os.path.join(output_dir, "sft_checkpoint")

    cfg = SFTConfig(
        output_dir                    = sft_out,
        num_train_epochs              = epochs,
        per_device_train_batch_size   = batch_size,
        per_device_eval_batch_size    = batch_size,
        gradient_accumulation_steps   = grad_accum,
        learning_rate                 = lr,
        lr_scheduler_type             = "cosine",
        bf16                          = True,
        tf32                          = True,
        gradient_checkpointing        = True,
        gradient_checkpointing_kwargs = {"use_reentrant": False},
        logging_steps                 = logging_steps,
        eval_strategy                 = "epoch",
        save_strategy                 = "epoch",
        save_total_limit              = 2,
        load_best_model_at_end        = True,
        metric_for_best_model         = "eval_loss",
        greater_is_better             = False,
        dataloader_num_workers        = 0,
        report_to                     = "none",
        remove_unused_columns         = False,
        dataset_kwargs                = {"skip_prepare_dataset": True},
        max_length                    = max_seq_len,
        fsdp                          = "full_shard auto_wrap" if use_fsdp else "",
        fsdp_config={
            "transformer_layer_cls_to_wrap": _get_fsdp_layer_cls(model),
            "use_orig_params":               True,
            "backward_prefetch":             "backward_pre",
            "forward_prefetch":              False,
        } if use_fsdp else None,
    )

    cb = StatsCallback()
    trainer = SFTTrainer(
        model            = model,
        args             = cfg,
        train_dataset    = train_ds,
        eval_dataset     = val_ds,
        processing_class = tokenizer,
        data_collator    = CompletionOnlyCollator(tokenizer, max_seq_len),
        callbacks        = [cb]
    )

    eff_batch = batch_size * grad_accum
    print(f"\n  epochs={epochs}  lr={lr}  batch={batch_size}×{grad_accum}={eff_batch}")
    trainer.train()
    trainer.save_model(sft_out)
    tokenizer.save_pretrained(sft_out)
    print(f"  [SFT] Checkpoint → {sft_out}")

    stats = cb.to_dict()
    save_training_stats({"stage": "sft", **stats}, sft_out)

    del trainer
    gc.collect()
    torch.cuda.empty_cache()

    if use_fsdp:
        # After FSDP training every Linear is wrapped with an FSDP shard module.
        # The only reliable way to get a clean, re-wrappable model is to
        # instantiate a fresh one and copy the saved adapter parameters into it.
        print("  [SFT] Creating fresh model and copying SFT parameters...")
        clean_model, _ = load_base_model(
            model_name   = model_name,
            lora_r       = lora_r,
            lora_alpha   = lora_alpha,
            lora_dropout = lora_dropout,
            use_fsdp     = False,   # plain model — DPOTrainer will wrap it
        )
        from peft import set_peft_model_state_dict
        adapter_sf  = os.path.join(sft_out, "adapter_model.safetensors")
        adapter_bin = os.path.join(sft_out, "adapter_model.bin")
        device = next(clean_model.parameters()).device
        if os.path.exists(adapter_sf):
            from safetensors.torch import load_file as _safe_load
            adapter_sd = _safe_load(adapter_sf, device=str(device))
        else:
            adapter_sd = torch.load(adapter_bin, map_location=device)
        set_peft_model_state_dict(clean_model, adapter_sd)
        _cast_to_bf16(clean_model)
        print(f"  [SFT] Fresh model ready → device={device}  dtype=bfloat16")
    else:
        clean_model = model
        if torch.cuda.is_available() and next(clean_model.parameters()).device.type == "cpu":
            print("  [SFT] Restoring model to GPU after load_best_model_at_end...")
            clean_model.cuda()

    return sft_out, stats, clean_model


# ── DPO training ──────────────────────────────────────────────────────────────

def run_dpo(
    model,
    tokenizer,
    train_ds,
    val_ds,
    output_dir: str,
    beta: float          = 0.25,
    epochs: int          = 1,
    batch_size: int      = 1,
    grad_accum: int      = 1,
    lr: float            = 5e-5,
    max_length: int      = 4096,
    warmup_ratio: float  = 0.1,
    logging_steps: int   = 5,
    use_fsdp: bool       = False,
    model_name: str      = "",
    lora_r: int          = 16,
    lora_alpha: int      = 32,
    lora_dropout: float  = 0.05,
) -> Tuple[str, Dict[str, Any], Any]:
    """
    DPO fine-tune starting from the SFT model.
    ref_model=None → TRL uses an implicit frozen copy of the policy at init time
    (i.e. the loaded SFT weights), which is exactly what we want.
    Returns (checkpoint_path, stats_dict, clean_model).
    """
    from trl import DPOTrainer, DPOConfig
    class FSDPSafeDPOTrainer(DPOTrainer):
        def _clip_grad_norm(self, model):
            if getattr(self, "is_fsdp_enabled", False) or self.args.fsdp:
                return None
            return super()._clip_grad_norm(model)

        def _get_grad_norm(self, model, grad_norm=None):
            if getattr(self, "is_fsdp_enabled", False) or self.args.fsdp:
                return None
            return super()._get_grad_norm(model, grad_norm=grad_norm)
    
    print(f"  [DPO] Model device: {next(model.parameters()).device}")
    dpo_out = os.path.join(output_dir, "dpo_checkpoint")
    cfg = DPOConfig(
        output_dir                    = dpo_out,
        num_train_epochs              = epochs,
        per_device_train_batch_size   = batch_size,
        per_device_eval_batch_size    = batch_size,
        gradient_accumulation_steps   = grad_accum,
        learning_rate                 = lr,
        lr_scheduler_type             = "cosine",
        warmup_ratio                  = warmup_ratio,
        bf16                          = True,
        tf32                          = True,
        gradient_checkpointing        =  False,
        gradient_checkpointing_kwargs = {"use_reentrant": False},
        logging_steps                 = logging_steps,
        eval_strategy                 = "epoch",
        save_strategy                 = "epoch",
        save_total_limit              = 2,
        load_best_model_at_end        = True,
        metric_for_best_model         = "eval_loss",
        greater_is_better             = False,
        dataloader_num_workers        = 0,
        report_to                     = "none",
        beta                          = beta,
        max_length                    = max_length,
        truncation_mode               = "keep_end",
        precompute_ref_log_probs      = True,
        precompute_ref_batch_size     = 1,
        remove_unused_columns         = False,
        padding_free                  = False,
        #max_grad_norm                 = 0.0,
        fsdp                          = "full_shard auto_wrap" if use_fsdp else "",
        fsdp_config={
            "transformer_layer_cls_to_wrap": _get_fsdp_layer_cls(model),
            "use_orig_params":               True,
            "backward_prefetch":             "backward_pre",
            "forward_prefetch":              False,
        } if use_fsdp else None,
    )

    cb = StatsCallback()
    trainer = FSDPSafeDPOTrainer(
        model            = model,
        ref_model        = None,   # implicit frozen SFT copy (TRL creates from model weights at init)
        args             = cfg,
        train_dataset    = train_ds,
        eval_dataset     = val_ds,
        processing_class = tokenizer,
        callbacks        = [cb],
    )

    if use_fsdp:
        _cast_to_bf16(trainer.model)
        if getattr(trainer, "ref_model", None) is not None:
            _cast_to_bf16(trainer.ref_model)

    eff_batch = batch_size * grad_accum
    print(f"\n  beta={beta}  epochs={epochs}  lr={lr}  batch={batch_size}×{grad_accum}={eff_batch}")
    print(f"  [DPO] Training model device: {next(trainer.model.parameters()).device}")

    model.config.use_cache = False
    model.enable_input_require_grads()

    trainer.train()
    trainer.save_model(dpo_out)
    tokenizer.save_pretrained(dpo_out)
    print(f"  [DPO] Checkpoint → {dpo_out}")

    stats = cb.to_dict()
    save_training_stats({"stage": "dpo", **stats}, dpo_out)

    del trainer
    gc.collect()
    torch.cuda.empty_cache()

    if use_fsdp:
        print("  [DPO] Creating fresh model and copying DPO parameters...")
        clean_model, _ = load_base_model(
            model_name   = model_name,
            lora_r       = lora_r,
            lora_alpha   = lora_alpha,
            lora_dropout = lora_dropout,
            use_fsdp     = False,
        )
        from peft import set_peft_model_state_dict
        adapter_sf  = os.path.join(dpo_out, "adapter_model.safetensors")
        adapter_bin = os.path.join(dpo_out, "adapter_model.bin")
        device = next(clean_model.parameters()).device
        if os.path.exists(adapter_sf):
            from safetensors.torch import load_file as _safe_load
            adapter_sd = _safe_load(adapter_sf, device=str(device))
        else:
            adapter_sd = torch.load(adapter_bin, map_location=device)
        set_peft_model_state_dict(clean_model, adapter_sd)
        _cast_to_bf16(clean_model)
        print(f"  [DPO] Fresh model ready → device={device}  dtype=bfloat16")
    else:
        clean_model = model
        if torch.cuda.is_available() and next(clean_model.parameters()).device.type == "cpu":
            clean_model.cuda()

    return dpo_out, stats, clean_model


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SFT → DPO fine-tuning from sonnet workflow dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--model", required=True, metavar="HF_MODEL_NAME")

    p.add_argument("--sft-data",      default="datasets/sonnet_workflow_v1/train_sft.jsonl")
    p.add_argument("--dpo-data",      default="datasets/sonnet_workflow_v1/train_dpo.jsonl")
    p.add_argument("--ranking-data",  default="datasets/sonnet_workflow_v1/train_ranking.jsonl")
    p.add_argument("--output-dir",    default="trained/")

    p.add_argument("--skip-sft",  action="store_true")
    p.add_argument("--skip-dpo",  action="store_true")
    p.add_argument("--sft-ckpt",  default=None,
                   help="Existing SFT adapter checkpoint (required with --skip-sft)")
    p.add_argument("--dry-run",   action="store_true")

    p.add_argument("--eval-only",     action="store_true")
    p.add_argument("--eval-sft-ckpt", default=None)
    p.add_argument("--eval-dpo-ckpt", default=None)

    p.add_argument("--sft-epochs",     type=int,   default=2)
    p.add_argument("--sft-lr",         type=float, default=1e-5)
    p.add_argument("--sft-batch",      type=int,   default=1)
    p.add_argument("--sft-grad-accum", type=int,   default=4)

    p.add_argument("--dpo-epochs",     type=int,   default=1)
    p.add_argument("--dpo-lr",         type=float, default=5e-5)
    p.add_argument("--dpo-batch",      type=int,   default=1)
    p.add_argument("--dpo-grad-accum", type=int,   default=4)
    p.add_argument("--dpo-beta",       type=float, default=0.1)

    p.add_argument("--lora-r",       type=int,   default=16)
    p.add_argument("--lora-alpha",   type=int,   default=32)
    p.add_argument("--lora-dropout", type=float, default=0.1)

    p.add_argument("--load-in-4bit", action="store_true")
    p.add_argument("--load-in-8bit", action="store_true")

    p.add_argument("--val-frac",      type=float, default=0.12)
    p.add_argument("--val-gen",       type=int,   default=20)
    p.add_argument("--val-dpo-pairs", type=int,   default=40)
    p.add_argument("--gen-batch",     type=int,   default=2)

    p.add_argument("--max-seq-len",     type=int, default=2500)
    p.add_argument("--dpo-max-seq-len", type=int, default=1500)#1500)#800
    p.add_argument("--dpo-min-samples", type=int, default=300,
                   help="Minimum DPO samples required. If fewer samples fit within "
                        "--dpo-max-seq-len, the shortfall is filled with truncated "
                        "oversized samples (sorted by score_gap desc). 0 = no minimum.")

    p.add_argument("--fsdp", action="store_true",
                   help="Enable FSDP. Launch with: accelerate launch --config_file configs/fsdp_config.yaml")

    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


# ── Eval-only entry point ─────────────────────────────────────────────────────

def run_eval_only(args) -> None:
    """Load saved adapters and run the full validation suite without training."""
    print("=" * 64)
    print("  Eval-only mode")
    print(f"  Model  : {args.model}")
    print(f"  Output : {args.output_dir}")
    print("=" * 64)

    sft_ckpt = args.eval_sft_ckpt or os.path.join(args.output_dir, "sft_checkpoint")
    dpo_ckpt = args.eval_dpo_ckpt or os.path.join(args.output_dir, "dpo_checkpoint")
    has_sft  = Path(sft_ckpt).exists()
    has_dpo  = Path(dpo_ckpt).exists()

    if not has_sft and not has_dpo:
        print(f"[error] No checkpoints found under {args.output_dir}")
        sys.exit(1)

    print(f"\n  SFT checkpoint : {sft_ckpt}  ({'found' if has_sft else 'NOT FOUND – skipped'})")
    print(f"  DPO checkpoint : {dpo_ckpt}  ({'found' if has_dpo else 'not found – skipped'})")

    print("\n[1/3] Loading validation data...")
    sft_train, sft_val = load_sft_data(args.sft_data, val_frac=args.val_frac, seed=args.seed)
    _, dpo_val          = load_dpo_data(args.dpo_data, val_frac=args.val_frac, seed=args.seed)
    ranking_data        = load_ranking_data(args.ranking_data)
    print(f"  SFT val={len(sft_val)}  DPO val={len(dpo_val)}")

    eval_report: Dict[str, Any] = {
        "model":          args.model,
        "output_dir":     args.output_dir,
        "sft_checkpoint": sft_ckpt if has_sft else None,
        "dpo_checkpoint": dpo_ckpt if has_dpo else None,
        "sft_validation": None,
        "dpo_validation": None,
        "reward_margin":  None,
    }

    sft_metrics: Optional[Dict[str, Any]] = None

    if has_sft:
        print(f"\n[2/3] Evaluating SFT checkpoint: {sft_ckpt}")
        model, tokenizer = load_model_from_checkpoint(
            model_name       = args.model,
            checkpoint_path  = sft_ckpt,
            load_in_4bit     = args.load_in_4bit,
            load_in_8bit     = args.load_in_8bit,
            use_fsdp         = args.fsdp,
        )
        sft_metrics = validate_checkpoint(
            model           = model,
            tokenizer       = tokenizer,
            sft_val_records = sft_val,
            ranking_data    = ranking_data,
            label           = "sft",
            max_gen         = args.val_gen,
            gen_batch       = args.gen_batch,
        )
        print_checkpoint_report(sft_metrics)
        eval_report["sft_validation"] = sft_metrics
        _save_eval_details(sft_metrics, sft_ckpt, "sft_eval_details.json")

        if has_dpo:
            del model
            gc.collect()
            torch.cuda.empty_cache()

    if has_dpo:
        print(f"\n[3/3] Evaluating DPO checkpoint: {dpo_ckpt}")
        model, tokenizer = load_model_from_checkpoint(
            model_name      = args.model,
            checkpoint_path = dpo_ckpt,
            load_in_4bit    = args.load_in_4bit,
            load_in_8bit    = args.load_in_8bit,
            use_fsdp        = args.fsdp,
        )
        dpo_metrics = validate_checkpoint(
            model           = model,
            tokenizer       = tokenizer,
            sft_val_records = sft_val,
            ranking_data    = ranking_data,
            label           = "dpo",
            max_gen         = args.val_gen,
            gen_batch       = args.gen_batch,
        )
        margin_metrics = validate_reward_margin(
            model           = model,
            tokenizer       = tokenizer,
            dpo_val_records = dpo_val,
            max_pairs       = args.val_dpo_pairs,
        )
        eval_report["dpo_validation"] = dpo_metrics
        eval_report["reward_margin"]  = margin_metrics
        _save_eval_details(dpo_metrics, dpo_ckpt, "dpo_eval_details.json")

        if sft_metrics is not None:
            print_dpo_comparison(sft_metrics, dpo_metrics, margin_metrics)
        else:
            print_checkpoint_report(dpo_metrics)

    save_report(eval_report, args.output_dir, name="eval_report.json")
    print(f"\n[done] Eval report → {args.output_dir}/eval_report.json")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # Silence non-rank-0 processes under FSDP
    if args.fsdp and int(os.environ.get("LOCAL_RANK", 0)) != 0:
        sys.stdout = open(os.devnull, "w")

    _check_deps()

    if args.eval_only:
        run_eval_only(args)
        return

    print("=" * 64)
    print("  SFT → DPO Fine-tuning Pipeline")
    print(f"  Model  : {args.model}")
    print(f"  Output : {args.output_dir}")
    print("=" * 64)

    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    is_main    = local_rank in (-1, 0)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    print("\n[1/6] Loading datasets...")  # steps: load / model / sft / sft-val / dpo / dpo-val
    sft_train, sft_val = load_sft_data(args.sft_data, val_frac=args.val_frac, seed=args.seed)
    dpo_train, dpo_val = load_dpo_data(args.dpo_data, val_frac=args.val_frac, seed=args.seed)
    ranking_data       = load_ranking_data(args.ranking_data)

    print(f"  SFT   train={len(sft_train):4d}  val={len(sft_val)}")
    print(f"  DPO   train={len(dpo_train):4d}  val={len(dpo_val)}")
    print(f"  Ranking index: {len(ranking_data)} queries")

    if args.dry_run:
        from transformers import AutoTokenizer
        trust = args.model in _TRUST_REMOTE
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=trust)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        def _tok_len(messages, add_gen_prompt=False):
            try:
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=add_gen_prompt
                )
            except Exception:
                text = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)
            return len(tokenizer(text, add_special_tokens=False)["input_ids"])

        def _stats(lengths, label):
            import statistics
            n   = len(lengths)
            srt = sorted(lengths)
            print(f"  {label:<22} n={n:5d}  mean={sum(lengths)/n:7.1f}  "
                  f"p50={srt[n//2]:5d}  p95={srt[min(n-1,int(n*0.95))]:5d}  "
                  f"p99={srt[min(n-1,int(n*0.99))]:5d}  max={max(lengths):5d}")

        print("\n[dry-run] Sample SFT record:")
        ex = sft_train[0]
        print(f"  query_id       : {ex['query_id']}")
        print(f"  positive_score : {ex['positive_score']:.2f}")
        print(f"  user msg[:200] : {ex['messages'][1]['content'][:200]}")

        print("\n[dry-run] SFT token length distribution:")
        sft_lens = [_tok_len(r["messages"]) for r in sft_train]
        _stats(sft_lens, "sft_full_sequence")

        print("\n[dry-run] DPO token length distribution:")
        dpo_prompt_lens   = [_tok_len(r["prompt"], add_gen_prompt=True) for r in dpo_train]
        dpo_chosen_lens   = [_tok_len(r["chosen"])   for r in dpo_train]
        dpo_rejected_lens = [_tok_len(r["rejected"])  for r in dpo_train]
        dpo_total_lens    = [p + max(c, rj) for p, c, rj in
                             zip(dpo_prompt_lens, dpo_chosen_lens, dpo_rejected_lens)]
        _stats(dpo_prompt_lens,  "dpo_prompt")
        _stats(dpo_chosen_lens,  "dpo_chosen")
        _stats(dpo_rejected_lens,"dpo_rejected")
        _stats(dpo_total_lens,   "dpo_prompt+max(c,r)")

        sft_truncated = sum(1 for l in sft_lens       if l > args.max_seq_len)
        dpo_truncated = sum(1 for l in dpo_total_lens if l > args.dpo_max_seq_len)
        print(f"\n[dry-run] Truncation at --max-seq-len={args.max_seq_len} / "
              f"--dpo-max-seq-len={args.dpo_max_seq_len}:")
        print(f"  SFT sequences truncated : {sft_truncated}/{len(sft_lens)} "
              f"({100*sft_truncated/len(sft_lens):.1f}%)")
        print(f"  DPO pairs truncated     : {dpo_truncated}/{len(dpo_total_lens)} "
              f"({100*dpo_truncated/len(dpo_total_lens):.1f}%)")
        print("\n[dry-run] Exiting without training.")
        return

    report: Dict[str, Any] = {
        "model":          args.model,
        "sft_data":       args.sft_data,
        "dpo_data":       args.dpo_data,
        "hyperparams": {
            "lora_r":       args.lora_r,
            "lora_alpha":   args.lora_alpha,
            "sft_epochs":   args.sft_epochs,
            "sft_lr":       args.sft_lr,
            "dpo_epochs":   args.dpo_epochs,
            "dpo_lr":       args.dpo_lr,
            "dpo_beta":     args.dpo_beta,
            "load_in_4bit": args.load_in_4bit,
            "load_in_8bit": args.load_in_8bit,
        },
        "sft_validation": None,
        "dpo_validation": None,
        "reward_margin":  None,
        "sft_checkpoint": None,
        "dpo_checkpoint": None,
    }

    # ── 2. Load model ONCE ────────────────────────────────────────────────────
    # A single base model + LoRA adapter is loaded here and reused for both
    # SFT and DPO.  No intermediate reloads.
    print(f"\n[2/6] Loading {args.model} with LoRA (r={args.lora_r}, α={args.lora_alpha})...")
    model, tokenizer = load_base_model(
        model_name   = args.model,
        lora_r       = args.lora_r,
        lora_alpha   = args.lora_alpha,
        lora_dropout = args.lora_dropout,
        load_in_4bit = args.load_in_4bit,
        load_in_8bit = args.load_in_8bit,
        use_fsdp     = args.fsdp,
    )

    # ── 3. SFT stage ──────────────────────────────────────────────────────────
    # Trains the LoRA adapter on SFT data and saves it to sft_checkpoint/.
    # The model object is updated in-place by SFTTrainer (load_best_model_at_end).
    sft_ckpt: Optional[str] = None
    if args.skip_sft:
        sft_ckpt = args.sft_ckpt or os.path.join(args.output_dir, "sft_checkpoint")
        if not Path(sft_ckpt).exists():
            print(f"[error] --skip-sft requires a checkpoint at {sft_ckpt}")
            sys.exit(1)
        print(f"\n[3/6] Skipping SFT — checkpoint: {sft_ckpt}")
    else:
        print("\n[3/6] SFT training...")
        sft_hf_train = build_sft_hf_dataset(sft_train, tokenizer)
        sft_hf_val   = build_sft_hf_dataset(sft_val,   tokenizer)
        sft_ckpt, sft_stats, model = run_sft(
            model        = model,
            tokenizer    = tokenizer,
            train_ds     = sft_hf_train,
            val_ds       = sft_hf_val,
            output_dir   = args.output_dir,
            epochs       = args.sft_epochs,
            batch_size   = args.sft_batch,
            grad_accum   = args.sft_grad_accum,
            lr           = args.sft_lr,
            max_seq_len  = args.max_seq_len,
            use_fsdp     = args.fsdp,
            model_name   = args.model,
            lora_r       = args.lora_r,
            lora_alpha   = args.lora_alpha,
            lora_dropout = args.lora_dropout,
        )
        if is_main:
            print_curve_summary(sft_stats, "SFT")
        # `model` is now the FSDP-unwrapped PEFT model with the best SFT weights
        # on GPU, ready to be handed directly to DPOTrainer.

    report["sft_checkpoint"] = sft_ckpt

    # ── 4. SFT validation ─────────────────────────────────────────────────────
    # run_sft under FSDP already returns a clean non-sharded model on cuda:0,
    # so generation works here even in a multi-GPU FSDP run (other ranks wait
    # at the DPO barrier below while rank 0 runs inference).
    sft_metrics: Optional[Dict[str, Any]] = None
    if is_main and not args.skip_sft:
        print("\n[4/6] SFT validation...")
        sft_metrics = validate_checkpoint(
            model           = model,
            tokenizer       = tokenizer,
            sft_val_records = sft_val,
            ranking_data    = ranking_data,
            label           = "sft",
            max_gen         = args.val_gen,
            gen_batch       = args.gen_batch,
        )
        print_checkpoint_report(sft_metrics)
        report["sft_validation"] = sft_metrics
        _save_eval_details(sft_metrics, sft_ckpt, "sft_eval_details.json")

    # ── 5. DPO stage ──────────────────────────────────────────────────────────
    # Continues training the same model object on DPO preference data.
    # DPOTrainer(ref_model=None) deep-copies the current model weights as the
    # frozen reference, so the SFT weights become the implicit reference policy.
    dpo_ckpt: Optional[str] = None
    if not args.skip_dpo:
        import torch.distributed as _dist
        if args.fsdp and _dist.is_initialized():
            _dist.barrier()

        _cast_to_bf16(model)
        model.enable_input_require_grads()
        model.train()
        print("\n[5/6] DPO training...")
        dpo_hf_train = build_dpo_hf_dataset_capped(dpo_train, tokenizer, max_seq_len=args.dpo_max_seq_len, min_samples=args.dpo_min_samples)
        dpo_hf_val   = build_dpo_hf_dataset_capped(dpo_val,   tokenizer, max_seq_len=args.dpo_max_seq_len, min_samples=0)
        dpo_ckpt, dpo_stats, model = run_dpo(
            model        = model,
            tokenizer    = tokenizer,
            train_ds     = dpo_hf_train,
            val_ds       = dpo_hf_val,
            output_dir   = args.output_dir,
            beta         = args.dpo_beta,
            epochs       = args.dpo_epochs,
            batch_size   = args.dpo_batch,
            grad_accum   = args.dpo_grad_accum,
            lr           = args.dpo_lr,
            max_length   = args.dpo_max_seq_len,
            use_fsdp     = args.fsdp,
            model_name   = args.model,
            lora_r       = args.lora_r,
            lora_alpha   = args.lora_alpha,
            lora_dropout = args.lora_dropout,
        )
        if is_main:
            print_curve_summary(dpo_stats, "DPO")
        report["dpo_checkpoint"] = dpo_ckpt
        # After run_dpo the in-memory model holds the best DPO weights.
    else:
        print("\n[5/6] Skipping DPO.")

    # ── 6. DPO validation ────────────────────────────────────────────────────
    # Same reasoning as SFT validation: run_dpo under FSDP returns a clean
    # non-sharded model, so rank 0 can run inference directly.
    if is_main and not args.skip_dpo and dpo_ckpt:
        print("\n[6/6] DPO validation...")
        dpo_metrics = validate_checkpoint(
            model           = model,
            tokenizer       = tokenizer,
            sft_val_records = sft_val,
            ranking_data    = ranking_data,
            label           = "dpo",
            max_gen         = args.val_gen,
            gen_batch       = args.gen_batch,
        )
        margin_metrics = validate_reward_margin(
            model           = model,
            tokenizer       = tokenizer,
            dpo_val_records = dpo_val,
            max_pairs       = args.val_dpo_pairs,
        )
        report["dpo_validation"] = dpo_metrics
        report["reward_margin"]  = margin_metrics
        _save_eval_details(dpo_metrics, dpo_ckpt, "dpo_eval_details.json")

        if sft_metrics is not None:
            print_dpo_comparison(sft_metrics, dpo_metrics, margin_metrics)
        else:
            print_checkpoint_report(dpo_metrics)
            rm = margin_metrics.get("reward_margin_mean")
            rp = margin_metrics.get("reward_margin_pos")
            print(f"\n  Reward margin (mean) : {rm:+.4f}" if rm is not None else "  Reward margin: N/A")
            print(f"  Chosen ranked higher : {rp:.1%}"    if rp is not None else "")
    elif args.skip_dpo and is_main:
        print("\n[6/6] DPO validation skipped (--skip-dpo).")

    # ── Save final report ─────────────────────────────────────────────────────
    # Only rank 0 writes the report — it's the only rank with validation data.
    if is_main:
        save_report(report, args.output_dir)
    print("\n[done] Pipeline complete.")
    print(f"  SFT checkpoint : {report['sft_checkpoint']}")
    if report["dpo_checkpoint"]:
        print(f"  DPO checkpoint : {report['dpo_checkpoint']}")
    print(f"  Report         : {args.output_dir}/training_report.json")


if __name__ == "__main__":
    main()


# ── Quick reference ───────────────────────────────────────────────────────────
# Single GPU:
# CUDA_VISIBLE_DEVICES=0 python scripts/train_sft_dpo.py \
#     --model meta-llama/Llama-3.2-3B-Instruct \
#     --output-dir /data/Kushal/AgenticWorkflow/trained/Llama-3.2-3B-Instruct \
#     > ./log/training/Llama-3.2-3B-Instruct.txt 2>&1
#
# Multi-GPU with FSDP:
# CUDA_VISIBLE_DEVICES=1,2,3,4,5,6 accelerate launch \
#     --config_file configs/fsdp_config.yaml \
#     scripts/train_sft_dpo.py \
#     --model meta-llama/Llama-3.2-3B-Instruct \
#     --output-dir /data/Kushal/AgenticWorkflow/trained/Llama-3.2-3B-Instruct \
#     --fsdp \
#     > ./log/training/Llama-3.2-3B-Instruct.txt 2>&1
#
# accelerate launch  --config_file configs/fsdp_config.yaml     scripts/train_sft_dpo.py     --model "Qwen/Qwen3.5-4B"  --output-dir /data/Kushal/AgenticWorkflow/trained/Qwen3.5-4B --fsdp     > ./log/training/Qwen3.5-4B.txt 2>&1  
# accelerate launch  --config_file configs/fsdp_config.yaml     scripts/train_sft_dpo.py     --model "meta-llama/Llama-3.2-3B-Instruct"  --output-dir /data/Kushal/AgenticWorkflow/trained/Llama-3.2-3B-Instruct --fsdp     > ./log/training/Llama-3.2-3B-Instruct.txt 2>&1  
# accelerate launch  --config_file configs/fsdp_config.yaml     scripts/train_sft_dpo.py     --model "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"  --output-dir /data/Kushal/AgenticWorkflow/trained/DeepSeek-R1-Distill-Qwen-1.5B --fsdp     > ./log/training/DeepSeek-R1-Distill-Qwen-1.5B.txt 2>&1  
# accelerate launch  --config_file configs/fsdp_config.yaml     scripts/train_sft_dpo.py     --model "HuggingFaceTB/SmolLM3-3B"  --output-dir /data/Kushal/AgenticWorkflow/trained/SmolLM3-3B --fsdp     > ./log/training/SmolLM3-3B.txt 2>&1  
# accelerate launch  --config_file configs/fsdp_config.yaml     scripts/train_sft_dpo.py     --model "microsoft/Phi-4-mini-instruct"  --output-dir /data/Kushal/AgenticWorkflow/trained/Phi-4-mini-instruct --fsdp     > ./log/training/Phi-4-mini-instruct.txt 2>&1  
#
# Skip SFT (already trained):
# python scripts/train_sft_dpo.py \
#     --model meta-llama/Llama-3.2-3B-Instruct \
#     --skip-sft \
#     --sft-ckpt /data/Kushal/AgenticWorkflow/trained/Llama-3.2-3B-Instruct/sft_checkpoint \
#     --output-dir /data/Kushal/AgenticWorkflow/trained/Llama-3.2-3B-Instruct
#
# Eval-only:
# python scripts/train_sft_dpo.py \
#     --model meta-llama/Llama-3.2-3B-Instruct \
#     --output-dir /data/Kushal/AgenticWorkflow/trained/Llama-3.2-3B-Instruct \
#     --eval-only
## Phi4 --> Phi3 ---> pip install transformers==4.53.3
## Phi4 --> Phi3 ---> pip install trl==0.20.0 
