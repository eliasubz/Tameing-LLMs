"""
Train runner: wraps train.py logic into a callable function so you can
run multiple configs sequentially.  Also includes all benchmark functions
(param table, memory wall, inference latency, summary bake-off) so
everything runs from a single file with shared config infrastructure.

Usage:
    # From Python / notebook:
    from train_runner import train
    train("config/train_shkspr_delta.py")

    # Full sweep + benchmarks from command line:
    python train_runner.py
    python train_runner.py --max_iters=2 --wandb_log=False   # smoke test
"""

import os
import sys
import gc
import time
import math
import pickle
from copy import deepcopy
from contextlib import nullcontext
from ast import literal_eval

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group


# ---------------------------------------------------------------------------
# Default config — same defaults as the original train.py
# ---------------------------------------------------------------------------
DEFAULT_CONFIG = dict(
    # which model architecture: "vanilla", "delta", or "delta_product"
    model_type="delta",
    # I/O
    out_dir="out-shakespeare",
    eval_interval=250,
    log_interval=5,
    eval_iters=200,
    eval_only=False,
    always_save_checkpoint=True,
    init_from="scratch",
    # wandb
    wandb_log=True,
    wandb_project="owt",
    wandb_run_name="gpt2",
    # data
    dataset="shakespeare_char",
    gradient_accumulation_steps=1,
    batch_size=64,
    block_size=256,
    # model
    n_layer=8,
    n_head=8,
    n_embd=384,
    dropout=0.2,
    bias=False,
    # adamw optimizer
    learning_rate=1e-3,
    max_iters=3000,
    weight_decay=1e-1,
    beta1=0.9,
    beta2=0.95,
    grad_clip=1.0,
    # lr decay
    decay_lr=True,
    warmup_iters=100,
    lr_decay_iters=3000,
    min_lr=1e-4,
    # DDP
    backend="nccl",
    # system
    device="cuda",
    dtype="bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16",
    compile=True,
)

# ---------------------------------------------------------------------------
# Shared model specifications for benchmarks — keeps arch hyper-params in
# one place so bench_params / bench_memory / bench_latency stay in sync
# with the training configs.
# ---------------------------------------------------------------------------
BENCH_MODELS = {
    "vanilla":       dict(n_layer=6, n_head=6, n_embd=384, dropout=0.0, bias=False),
    "delta":         dict(n_layer=6, n_head=6, n_embd=372, dropout=0.0, bias=False),
    "delta_product": dict(n_layer=6, n_head=2, n_embd=228, dropout=0.0, bias=False),
    "nsa":           dict(n_layer=6, n_head=6, n_embd=384, dropout=0.0, bias=False),
}
BENCH_VOCAB_SIZE  = 65     # shakespeare_char
BENCH_BLOCK_SIZE  = 256
BENCH_WANDB_PROJECT = "transformers"


def _import_model_module(model_type):
    """Dynamically import the right (GPTConfig, GPT) pair."""
    if model_type == "vanilla":
        from model import GPTConfig, GPT
    elif model_type == "delta":
        from model_delta import GPTConfig, GPT
    elif model_type == "delta_product":
        from model_gated_delta_product import GPTConfig, GPT
    elif model_type == "nsa":
        from model_nsa import GPTConfig, GPT
    else:
        raise ValueError(f"Unknown model_type: {model_type!r}. "
                         f"Choose from 'vanilla', 'delta', 'delta_product', 'nsa'.")
    return GPTConfig, GPT


def _load_config(config_file=None, overrides=None):
    """
    Build a config dict by:
      1. Starting from DEFAULT_CONFIG
      2. exec-ing the config_file (if provided) — its variables override defaults
      3. Applying key=value overrides dict on top
    """
    cfg = deepcopy(DEFAULT_CONFIG)

    if config_file is not None:
        # exec the config file so that bare assignments like `n_layer = 6`
        # land directly into cfg. We pass cfg as both globals and locals
        # so that simple assignments are written back into it.
        with open(config_file) as f:
            exec(compile(f.read(), config_file, "exec"), cfg, cfg)
        # clean up dunders that exec puts in
        cfg = {k: v for k, v in cfg.items() if not k.startswith("__")}

    if overrides:
        cfg.update(overrides)

    return cfg


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------
def train(config_file=None, overrides=None):
    """
    Run a full training loop.

    Args:
        config_file: path to a config .py file (e.g. "config/train_shkspr_delta.py").
                     Variables defined in it override DEFAULT_CONFIG.
        overrides:   dict of extra overrides applied on top, e.g. {"max_iters": 500}.
    Returns:
        dict with final train_loss, val_loss, and out_dir.
    """
    C = _load_config(config_file, overrides)

    # ---- unpack everything into local vars (mirrors original train.py) ----
    model_type              = C["model_type"]
    out_dir                 = C["out_dir"]
    eval_interval           = C["eval_interval"]
    log_interval            = C["log_interval"]
    eval_iters              = C["eval_iters"]
    eval_only               = C["eval_only"]
    always_save_checkpoint  = C["always_save_checkpoint"]
    init_from               = C["init_from"]
    wandb_log               = C["wandb_log"]
    wandb_project           = C["wandb_project"]
    wandb_run_name          = C["wandb_run_name"]
    dataset                 = C["dataset"]
    gradient_accumulation_steps = C["gradient_accumulation_steps"]
    batch_size              = C["batch_size"]
    block_size              = C["block_size"]
    n_layer                 = C["n_layer"]
    n_head                  = C["n_head"]
    n_embd                  = C["n_embd"]
    dropout                 = C["dropout"]
    bias                    = C["bias"]
    learning_rate           = C["learning_rate"]
    max_iters               = C["max_iters"]
    weight_decay            = C["weight_decay"]
    beta1                   = C["beta1"]
    beta2                   = C["beta2"]
    grad_clip               = C["grad_clip"]
    decay_lr                = C["decay_lr"]
    warmup_iters            = C["warmup_iters"]
    lr_decay_iters          = C["lr_decay_iters"]
    min_lr                  = C["min_lr"]
    backend                 = C["backend"]
    device                  = C["device"]
    dtype                   = C["dtype"]
    do_compile              = C["compile"]

    # keep a serialisable copy for checkpoint / wandb
    config = {k: v for k, v in C.items()
              if isinstance(v, (int, float, bool, str))}

    # ---- import the right model ----
    GPTConfig, GPT = _import_model_module(model_type)


    print(f"\n{'='*60}")
    print(f"  Training: model_type={model_type}  run={wandb_run_name}")
    print(f"{'='*60}\n")

    # ---- DDP setup ----
    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        init_process_group(backend=backend)
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
        seed_offset = ddp_rank
        assert gradient_accumulation_steps % ddp_world_size == 0
        gradient_accumulation_steps //= ddp_world_size
    else:
        master_process = True
        seed_offset = 0
        ddp_world_size = 1

    tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
    print(f"tokens per iteration will be: {tokens_per_iter:,}")

    if master_process:
        os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(1337 + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device_type = "cuda" if "cuda" in device else "cpu"
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16,
               "float16": torch.float16}[dtype]
    ctx = (nullcontext() if device_type == "cpu"
           else torch.amp.autocast(device_type=device_type, dtype=ptdtype))

    # ---- data loader ----
    data_dir = os.path.join("data", dataset)

    def get_batch(split):
        if split == "train":
            data = np.memmap(os.path.join(data_dir, "train.bin"), dtype=np.uint16, mode="r")
        else:
            data = np.memmap(os.path.join(data_dir, "val.bin"), dtype=np.uint16, mode="r")
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
        if device_type == "cuda":
            x, y = (x.pin_memory().to(device, non_blocking=True),
                     y.pin_memory().to(device, non_blocking=True))
        else:
            x, y = x.to(device), y.to(device)
        return x, y

    # ---- vocab ----
    iter_num = 0
    best_val_loss = 1e9

    meta_path = os.path.join(data_dir, "meta.pkl")
    meta_vocab_size = None
    if os.path.exists(meta_path):
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        meta_vocab_size = meta["vocab_size"]
        print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

    # ---- model init ----
    model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd,
                      block_size=block_size, bias=bias, vocab_size=None,
                      dropout=dropout)

    if init_from == "scratch":
        print("Initializing a new model from scratch")
        if meta_vocab_size is None:
            print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
        model_args["vocab_size"] = meta_vocab_size if meta_vocab_size is not None else 50304
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
    elif init_from == "resume":
        print(f"Resuming training from {out_dir}")
        ckpt_path = os.path.join(out_dir, "ckpt.pt")
        checkpoint = torch.load(ckpt_path, map_location=device)
        checkpoint_model_args = checkpoint["model_args"]
        for k in ["n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size"]:
            model_args[k] = checkpoint_model_args[k]
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
        state_dict = checkpoint["model"]
        unwanted_prefix = "_orig_mod."
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        iter_num = checkpoint["iter_num"]
        best_val_loss = checkpoint["best_val_loss"]
    elif init_from.startswith("gpt2"):
        print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
        override_args = dict(dropout=dropout)
        model = GPT.from_pretrained(init_from, override_args)
        for k in ["n_layer", "n_head", "n_embd", "block_size", "bias", "vocab_size"]:
            model_args[k] = getattr(model.config, k)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_params = trainable_params / 1e6  # Return in Millions for easy reading

    print(f"Model: {model_type:12} | Trainable Params: {trainable_params:.2f}M")


    if block_size < model.config.block_size:
        model.crop_block_size(block_size)
        model_args["block_size"] = block_size
    model.to(device)

    scaler = torch.amp.GradScaler(enabled=(dtype == "float16"))
    optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
    if init_from == "resume":
        optimizer.load_state_dict(checkpoint["optimizer"])
        checkpoint = None

    if do_compile:
        print("compiling the model... (takes a ~minute)")
        model = torch.compile(model)

    if ddp:
        model = DDP(model, device_ids=[int(device.split(":")[-1])])

    raw_model = model.module if ddp else model

    # ---- helpers ----
    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split in ["train", "val"]:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = get_batch(split)
                with ctx:
                    logits, loss, _ = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out

    def get_lr(it):
        if it < warmup_iters:
            return learning_rate * (it + 1) / (warmup_iters + 1)
        if it > lr_decay_iters:
            return min_lr
        decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (learning_rate - min_lr)

    # ---- wandb ----
    if wandb_log and master_process:
        import wandb
        wandb.init(project=wandb_project, name=wandb_run_name, config=config)

    # ---- FLOPs bookkeeping (for PPL-vs-FLOPs figure) ----
    N_params  = raw_model.get_num_params()
    cfg_model = raw_model.config
    _L, _H, _Q, _T = (cfg_model.n_layer, cfg_model.n_head,
                       cfg_model.n_embd // cfg_model.n_head, cfg_model.block_size)
    flops_per_token = 6 * N_params + 12 * _L * _H * _Q * _T
    flops_per_iter  = flops_per_token * _T * (batch_size * gradient_accumulation_steps)
    cumulative_flops = 0.0

    # ---- training loop ----
    X, Y = get_batch("train")
    t0 = time.time()
    local_iter_num = 0
    running_mfu = -1.0

    while True:
        lr = get_lr(iter_num) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        if iter_num % eval_interval == 0 and master_process:
            losses = estimate_loss()
            train_ppl = math.exp(losses["train"]) if losses["train"] < 20 else float("inf")
            val_ppl   = math.exp(losses["val"])   if losses["val"]   < 20 else float("inf")
            print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, "
                  f"train ppl {train_ppl:.2f}, val ppl {val_ppl:.2f}")
            if wandb_log:
                wandb.log({
                    "iter": iter_num,
                    "train/loss": losses["train"],
                    "val/loss": losses["val"],
                    "train/perplexity": train_ppl,
                    "val/perplexity": val_ppl,
                    "lr": lr,
                    "mfu": running_mfu * 100,
                    "cumulative_flops": cumulative_flops,
                })
            if losses["val"] < best_val_loss or always_save_checkpoint:
                best_val_loss = losses["val"]
                if iter_num > 0:
                    ckpt = {
                        "model": raw_model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "model_args": model_args,
                        "iter_num": iter_num,
                        "best_val_loss": best_val_loss,
                        "config": config,
                    }
                    print(f"saving checkpoint to {out_dir}")
                    torch.save(ckpt, os.path.join(out_dir, "ckpt.pt"))
        if iter_num == 0 and eval_only:
            break

        for micro_step in range(gradient_accumulation_steps):
            if ddp:
                model.require_backward_grad_sync = (
                    micro_step == gradient_accumulation_steps - 1
                )
            with ctx:
                logits, loss, _ = model(X, Y)
                loss = loss / gradient_accumulation_steps
            X, Y = get_batch("train")
            scaler.scale(loss).backward()

        if grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if iter_num % log_interval == 0 and master_process:
            lossf = loss.item() * gradient_accumulation_steps
            if local_iter_num >= 5:
                mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
                running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
            print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")

        cumulative_flops += flops_per_iter
        iter_num += 1
        local_iter_num += 1

        if iter_num > max_iters:
            break

    # ---- cleanup ----
    if ddp:
        destroy_process_group()

    try:
        final_train = float(losses["train"])
    except NameError:
        final_train = None

    final_losses = {"train_loss": final_train,
                    "val_loss": float(best_val_loss),
                    "out_dir": out_dir}

    if wandb_log and master_process:
        wandb.finish()

    print(f"\nDone: {wandb_run_name}  best_val_loss={best_val_loss:.4f}\n")
    return final_losses


# ===================================================================
# BENCHMARK FUNCTIONS
# ===================================================================

def bench_params(wandb_log=True):
    """Table 5 — Param-Matching Ablation.  Proves all 3 models have
    (nearly) identical trainable parameter counts."""

    rows = []
    for model_name, spec in BENCH_MODELS.items():
        GPTConfig, GPT = _import_model_module(model_name)
        cfg = GPTConfig(block_size=BENCH_BLOCK_SIZE, vocab_size=BENCH_VOCAB_SIZE, **spec)
        m = GPT(cfg)
        total = sum(p.numel() for p in m.parameters() if p.requires_grad)
        rows.append(dict(model_type=model_name, **spec,
                         total_trainable_params=total,
                         params_M=round(total / 1e6, 3)))
        del m

    print("\n" + "=" * 75)
    print(f"{'Model':<18} {'n_layer':>8} {'n_embd':>8} {'n_head':>8} {'Params (M)':>12}")
    print("-" * 75)
    for r in rows:
        print(f"{r['model_type']:<18} {r['n_layer']:>8} {r['n_embd']:>8} "
              f"{r['n_head']:>8} {r['params_M']:>12.3f}")
    print("=" * 75)

    if wandb_log:
        import wandb
        wandb.init(project=BENCH_WANDB_PROJECT, name="param_matching_table",
                   config={"block_size": BENCH_BLOCK_SIZE, "vocab_size": BENCH_VOCAB_SIZE})
        tbl = wandb.Table(
            columns=["model_type", "n_layer", "n_embd", "n_head",
                      "total_trainable_params", "params_M"],
            data=[[r["model_type"], r["n_layer"], r["n_embd"], r["n_head"],
                   r["total_trainable_params"], r["params_M"]] for r in rows])
        wandb.log({"params/matching_table": tbl})
        wandb.finish()
    return rows


def bench_memory(wandb_log=True, context_lengths=None, batch_size=4):
    """Figure 2 — Memory Wall (OOM).  Sweep context lengths and record
    peak VRAM after a single fwd+bwd pass."""

    if context_lengths is None:
        context_lengths = [256, 512, 1024, 2048, 4096, 8192, 16384]

    device = "cuda"
    dtstr  = "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16"
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16,
               "float16": torch.float16}[dtstr]

    if wandb_log:
        import wandb
        wandb.init(project=BENCH_WANDB_PROJECT, name="bench_memory_wall",
                   config={"context_lengths": context_lengths, "batch_size": batch_size})

    results = []
    for model_name, spec in BENCH_MODELS.items():
        GPTConfig, GPT = _import_model_module(model_name)
        for ctx_len in context_lengths:
            torch.cuda.empty_cache(); gc.collect()
            torch.cuda.reset_peak_memory_stats()

            try:
                cfg = GPTConfig(block_size=ctx_len, vocab_size=BENCH_VOCAB_SIZE, **spec)
                m = GPT(cfg).to(device); m.train()
                x = torch.randint(0, BENCH_VOCAB_SIZE, (batch_size, ctx_len), device=device)
                y = torch.randint(0, BENCH_VOCAB_SIZE, (batch_size, ctx_len), device=device)
                with torch.amp.autocast(device_type="cuda", dtype=ptdtype):
                    logits, loss, _ = m(x, y)
                loss.backward()
                peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
                oom = False
                del logits, loss, m, x, y
                torch.cuda.empty_cache(); gc.collect()
            except torch.cuda.OutOfMemoryError:
                peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
                oom = True
                torch.cuda.empty_cache(); gc.collect()

            row = dict(model_type=model_name, context_length=ctx_len,
                       peak_vram_mb=round(peak_mb, 1), oom=oom)
            results.append(row)
            print(f"  {model_name:16} ctx={ctx_len:6}  →  "
                  f"{'OOM' if oom else f'{peak_mb:.1f} MB'}")

            if wandb_log:
                import wandb
                wandb.log({"memory/model_type": model_name,
                           "memory/context_length": ctx_len,
                           "memory/peak_vram_mb": peak_mb,
                           "memory/oom": int(oom)})

    # summary
    print("\n" + "=" * 70)
    print(f"{'Model':<18} {'CtxLen':>8} {'Peak VRAM (MB)':>16} {'OOM':>5}")
    print("-" * 70)
    for r in results:
        print(f"{r['model_type']:<18} {r['context_length']:>8} "
              f"{r['peak_vram_mb']:>16.1f} {'YES' if r['oom'] else 'no':>5}")
    print("=" * 70)

    if wandb_log:
        import wandb
        tbl = wandb.Table(columns=["model_type", "context_length", "peak_vram_mb", "oom"],
                          data=[[r["model_type"], r["context_length"],
                                 r["peak_vram_mb"], r["oom"]] for r in results])
        wandb.log({"memory/summary_table": tbl})
        wandb.finish()
    return results


def _time_vanilla_gen(model, prompt, n_warm, n_meas, ptdtype):
    """Measure per-token latency for vanilla GPT (full re-attention)."""
    bs = model.config.block_size
    idx = prompt.clone()
    for _ in range(n_warm):
        ic = idx if idx.size(1) <= bs else idx[:, -bs:]
        with torch.amp.autocast(device_type="cuda", dtype=ptdtype):
            lo, _, _ = model(ic)
        idx = torch.cat([idx, lo[:, -1:].argmax(-1)], 1)
    torch.cuda.synchronize()
    times = []
    for _ in range(n_meas):
        ic = idx if idx.size(1) <= bs else idx[:, -bs:]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.amp.autocast(device_type="cuda", dtype=ptdtype):
            lo, _, _ = model(ic)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)
        idx = torch.cat([idx, lo[:, -1:].argmax(-1)], 1)
    return sum(times) / len(times)


def _time_delta_gen(model, prompt, n_warm, n_meas, ptdtype):
    """Measure per-token latency for Delta / DeltaProduct (recurrent state)."""
    idx = prompt.clone()
    with torch.amp.autocast(device_type="cuda", dtype=ptdtype):
        lo, _, st = model(idx, states=None, pos_offset=0)
    po = idx.size(1)
    nt = lo[:, -1:].argmax(-1)
    idx = torch.cat([idx, nt], 1)
    for _ in range(n_warm):
        with torch.amp.autocast(device_type="cuda", dtype=ptdtype):
            lo, _, st = model(nt, states=st, pos_offset=po)
        po += 1; nt = lo[:, -1:].argmax(-1)
        idx = torch.cat([idx, nt], 1)
    torch.cuda.synchronize()
    times = []
    for _ in range(n_meas):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.amp.autocast(device_type="cuda", dtype=ptdtype):
            lo, _, st = model(nt, states=st, pos_offset=po)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)
        po += 1; nt = lo[:, -1:].argmax(-1)
        idx = torch.cat([idx, nt], 1)
    return sum(times) / len(times)


def bench_latency(wandb_log=True, prompt_lengths=None, n_warm=3, n_meas=10):
    """Figure 4 — Inference Latency.  Time per token vs prompt length."""

    if prompt_lengths is None:
        prompt_lengths = [128, 256, 512, 1024]

    device = "cuda"
    dtstr  = "bfloat16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "float16"
    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16,
               "float16": torch.float16}[dtstr]

    if wandb_log:
        import wandb
        wandb.init(project=BENCH_WANDB_PROJECT, name="bench_inference_latency",
                   config={"prompt_lengths": prompt_lengths,
                           "n_warm": n_warm, "n_meas": n_meas})
    results = []
    for model_name, spec in BENCH_MODELS.items():
        GPTConfig, GPT = _import_model_module(model_name)
        for plen in prompt_lengths:
            torch.cuda.empty_cache(); gc.collect()
            bs = min(max(plen * 2, plen + n_warm + n_meas + 16), 16384)
            try:
                cfg = GPTConfig(block_size=bs, vocab_size=BENCH_VOCAB_SIZE, **spec)
                m = GPT(cfg).to(device).eval()
                prompt = torch.randint(0, BENCH_VOCAB_SIZE, (1, plen), device=device)
                if model_name in ("vanilla", "nsa"):
                    avg_ms = _time_vanilla_gen(m, prompt, n_warm, n_meas, ptdtype)
                else:
                    avg_ms = _time_delta_gen(m, prompt, n_warm, n_meas, ptdtype)
                oom = False
                del m; torch.cuda.empty_cache(); gc.collect()
            except torch.cuda.OutOfMemoryError:
                avg_ms = float("nan"); oom = True
                torch.cuda.empty_cache(); gc.collect()

            row = dict(model_type=model_name, prompt_length=plen,
                       avg_ms_per_token=round(avg_ms, 3) if not oom else None, oom=oom)
            results.append(row)
            print(f"  {model_name:16} prompt={plen:6}  →  "
                  f"{'OOM' if oom else f'{avg_ms:.3f} ms'}")

            if wandb_log:
                import wandb
                d = {"latency/model_type": model_name,
                     "latency/prompt_length": plen, "latency/oom": int(oom)}
                if not oom:
                    d["latency/ms_per_token"] = avg_ms
                wandb.log(d)

    # summary
    print("\n" + "=" * 65)
    print(f"{'Model':<18} {'Prompt':>8} {'ms/token':>12} {'OOM':>5}")
    print("-" * 65)
    for r in results:
        ms = f"{r['avg_ms_per_token']:.3f}" if r['avg_ms_per_token'] is not None else "—"
        print(f"{r['model_type']:<18} {r['prompt_length']:>8} {ms:>12} "
              f"{'YES' if r['oom'] else 'no':>5}")
    print("=" * 65)

    if wandb_log:
        import wandb
        tbl = wandb.Table(
            columns=["model_type", "prompt_length", "ms_per_token", "oom"],
            data=[[r["model_type"], r["prompt_length"],
                   r["avg_ms_per_token"], r["oom"]] for r in results])
        wandb.log({"latency/summary_table": tbl})
        wandb.finish()
    return results


def bench_summary(wandb_log=True, wandb_entity=None):
    """Table 3 + Figure 6 — Final Bake-Off and PPL vs FLOPs (log-log).
    Pulls data from wandb API after training runs + memory benchmark."""

    try:
        import wandb
    except ImportError:
        print("wandb not installed — cannot fetch run data"); return

    api  = wandb.Api()
    path = f"{wandb_entity}/{BENCH_WANDB_PROJECT}" if wandb_entity else BENCH_WANDB_PROJECT

    # ── fetch training runs ─────────────────────────────────────────────
    print("Fetching training runs from wandb...")
    training_runs = []
    for run in api.runs(path):
        if run.name.startswith("bench_"):
            continue
        s = run.summary._json_dict
        if "val/loss" not in s:
            continue
        c = run.config
        vl = s.get("val/loss")
        vp = s.get("val/perplexity") or (math.exp(vl) if vl and vl < 20 else None)
        training_runs.append(dict(
            run_name=run.name, model_type=c.get("model_type", "unknown"),
            val_loss=vl, val_ppl=vp,
            params_M=c.get("trainable_params_M"),
            learning_rate=c.get("learning_rate"),
            cumulative_flops=s.get("cumulative_flops")))

    if not training_runs:
        print("No training runs found. Run the full training sweep first."); return
    print(f"  Found {len(training_runs)} training run(s)")

    # ── fetch OOM thresholds ────────────────────────────────────────────
    print("Fetching OOM thresholds...")
    oom_thresh = {}
    for run in api.runs(path, filters={"display_name": "bench_memory_wall"}):
        for row in run.scan_history(keys=["memory/model_type",
                                           "memory/context_length",
                                           "memory/oom"]):
            mt, ctx, oom = row.get("memory/model_type"), row.get("memory/context_length"), row.get("memory/oom")
            if mt and ctx and not oom:
                oom_thresh[mt] = max(oom_thresh.get(mt, 0), ctx)

    # ── Table 3: Final Bake-Off ─────────────────────────────────────────
    best = {}
    for r in training_runs:
        mt = r["model_type"]
        if mt not in best or (r["val_loss"] is not None and r["val_loss"] < best[mt]["val_loss"]):
            best[mt] = r

    print("\n" + "=" * 85)
    print("TABLE 3 — THE FINAL BAKE-OFF")
    print("=" * 85)
    print(f"{'Model':<18} {'Best Run':<26} {'Val Loss':>10} {'Val PPL':>10} "
          f"{'Params(M)':>10} {'OOM Thresh':>10}")
    print("-" * 85)
    for mt in ["vanilla", "delta", "delta_product", "nsa"]:
        if mt not in best:
            continue
        r = best[mt]
        oom = oom_thresh.get(mt, "—")
        print(f"{mt:<18} {r['run_name']:<26} "
              f"{r['val_loss']:>10.4f} {r['val_ppl']:>10.2f} "
              f"{str(r.get('params_M','—')):>10} {str(oom):>10}")
    print("=" * 85)

    # ── Figure 6: PPL vs FLOPs ──────────────────────────────────────────
    print("\nFetching PPL vs FLOPs data...")
    series = {}
    for run in api.runs(path):
        if run.name.startswith("bench_"):
            continue
        c = run.config
        mt, lr = c.get("model_type", "?"), c.get("learning_rate", 0)
        for row in run.scan_history(keys=["cumulative_flops", "val/perplexity",
                                           "val/loss"]):
            flops = row.get("cumulative_flops")
            ppl   = row.get("val/perplexity")
            if ppl is None:
                vl = row.get("val/loss")
                if vl and vl < 20:
                    ppl = math.exp(vl)
            if flops and ppl:
                key = f"{mt}_lr{lr:.1e}" if lr else mt
                series.setdefault(key, []).append((flops, ppl))
    print(f"  Found {len(series)} series")

    # ── push to wandb ───────────────────────────────────────────────────
    if wandb_log:
        wandb.init(project=BENCH_WANDB_PROJECT, name="summary_bakeoff_ppl_flops",
                   config={"description": "Table 3 + Figure 6"})
        tbl_cols = ["model_type", "run_name", "val_loss", "val_ppl",
                     "params_M", "oom_threshold"]
        tbl_data = [[mt, best[mt]["run_name"], best[mt]["val_loss"],
                      best[mt]["val_ppl"], best[mt].get("params_M"),
                      oom_thresh.get(mt)]
                     for mt in ["vanilla", "delta", "delta_product", "nsa"] if mt in best]
        wandb.log({"bakeoff/summary_table": wandb.Table(columns=tbl_cols, data=tbl_data)})

        if series:
            pf = []
            for label, pts in series.items():
                pts.sort()
                for f, p in pts:
                    pf.append([label, f, p,
                               math.log10(f) if f > 0 else 0,
                               math.log10(p) if p > 0 else 0])
            wandb.log({"ppl_vs_flops/data": wandb.Table(
                columns=["series", "cumulative_flops", "ppl",
                          "log10_flops", "log10_ppl"], data=pf)})
        wandb.finish()
    print("Done.")


# ===================================================================
# __main__
# ===================================================================
if __name__ == "__main__":
    print("Main is starting")
    # ---- sweep config ----
    models_to_test = [
        "config/train_shakespeare_char.py",   # Vanilla
        "config/train_shkspr_ungated_delta.py",        # Delta
        # "config/train_shkspr_delta_prod.py",   # Delta Product
        "config/train_shkspr_nsa.py",          # Native Sparse Attention
    ]
    learning_rates = [5e-4, 1e-3, 3e-3]

    # ---- parse CLI overrides (--key=value) applied on top of every run ----
    extra_overrides = {}
    for arg in sys.argv[1:]:
        if arg.startswith("--") and "=" in arg:
            key, val = arg.split("=", 1)
            key = key[2:]
            try:
                val = literal_eval(val)
            except (SyntaxError, ValueError):
                pass
            extra_overrides[key] = val

    use_wandb = extra_overrides.get("wandb_log", True)

    # ---- run the grid: models × learning_rates ----
    for lr in learning_rates:
        for config_path in models_to_test:
            base_name = config_path.split('_')[-1].replace('.py', '')
            if 'char' in config_path:
                base_name = 'vanilla'
            elif 'nsa' in config_path:
                base_name = 'nsa'
            run_id = f"{base_name}_lr{lr:.1e}"
            overrides = {
                "learning_rate": lr,
                "wandb_run_name": run_id,
                "out_dir": f"out-{run_id}",
            }
            overrides.update(extra_overrides)
            print(f"\nSTARTING RUN: {run_id}")
            train(config_file=config_path, overrides=overrides)

    # ---- benchmarks ----
    print("\n" + "=" * 60)
    print("  RUNNING BENCHMARKS")
    print("=" * 60 + "\n")

    print(">>> Benchmark: Param-Matching Table (Table 5)")
    bench_params(wandb_log=use_wandb)

    print("\n>>> Benchmark: Memory Wall (Figure 2)")
    bench_memory(wandb_log=use_wandb)

    print("\n>>> Benchmark: Inference Latency (Figure 4)")
    bench_latency(wandb_log=use_wandb)

    print("\n>>> Benchmark: Bake-Off + PPL vs FLOPs (Table 3 / Figure 6)")
    bench_summary(wandb_log=use_wandb)

    print("\n" + "=" * 60)
    print("  ALL DONE — training + benchmarks complete")
    print("=" * 60)
