# Training Recipes for Gemma-4 Post-Training (Phase 3)

This directory provides concrete, executable training recipes, configuration templates, and launcher scripts for training the two LoRA adapters specified in the SWE agent architecture:

1. **`tool_lora`**: Localization sub-agent (`code_analyzer.yaml`).
   - Input: Problem statement / issue title.
   - Output: Ranked symbol localization report.
   - Target dataset: `out/sft/tool_lora_train.jsonl` (validation: `out/sft/tool_lora_val.jsonl`).
   
2. **`main_lora`**: Primary coder agent (`agent.yaml`).
   - Input: Problem statement + sub-agent report + repo context.
   - Output: Full explore $\to$ read $\to$ edit/write $\to$ verify $\to$ submit tool-calling loop.
   - Target dataset: `out/sft/main_lora_train.jsonl` (validation: `out/sft/main_lora_val.jsonl`).

---

## Hardware Targets & Frameworks

- **Base Model**: `google/gemma-4-31b-it` (or quantized QAT checkpoint `gemma-4-31b-it-qat-w4a16-ct`).
- **Recommended Environments**:
  - Multi-GPU (A100/H100 80GB) using Hugging Face `peft` + `trl` (`SFTTrainer`) with DeepSpeed Stage 2/3 or FSDP.
  - Single/Dual GPU (e.g. 1-2x A100/A6000) using `unsloth` 4-bit QLoRA.

## Hyperparameter Profiles

| Parameter | `tool_lora` (Analyzer) | `main_lora` (Coder Agent) |
|---|---|---|
| **Base Model** | `gemma-4-31b-it` | `gemma-4-31b-it` |
| **LoRA Rank ($r$)** | 16 | 64 |
| **LoRA Alpha ($\alpha$)** | 32 | 128 |
| **LoRA Target Modules** | `q_proj`, `v_proj`, `k_proj`, `o_proj` | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| **LoRA Dropout** | 0.05 | 0.05 |
| **Max Sequence Length** | 2048 | 8192 (or 16384 with packing / flash-attn) |
| **Learning Rate** | 2e-4 (cosine decay) | 1e-4 (cosine decay) |
| **Effective Batch Size** | 16 | 32 |
| **Epochs** | 3 | 3 |
| **Warmup Ratio** | 0.05 | 0.03 |
| **Weight Decay** | 0.01 | 0.01 |
