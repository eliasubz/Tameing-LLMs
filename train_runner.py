"""
Train runner: wraps train.py logic into a callable function so you can
run multiple configs sequentially.

Usage:
    # From Python / notebook:
    from train_runner import train
    train("config/train_shkspr_delta.py")
    train("config/train_shkspr_delta_prod.py")

    # From command line (behaves like train.py):
    python train_runner.py config/train_shkspr_delta.py
    python train_runner.py config/train_shkspr_delta_prod.py --max_iters=3000
"""

import os
import time
import math
import pickle
from copy import deepcopy
from contextlib import nullcontext
from ast import literal_eval
import sys

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


def _import_model_module(model_type):
    """Dynamically import the right (GPTConfig, GPT) pair."""
    if model_type == "vanilla":
        from model import GPTConfig, GPT
    elif model_type == "delta":
        from model_delta import GPTConfig, GPT
    elif model_type == "delta_product":
        from model_gated_delta_product import GPTConfig, GPT
    else:
        raise ValueError(f"Unknown model_type: {model_type!r}. "
                         f"Choose from 'vanilla', 'delta', 'delta_product'.")
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
            print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            if wandb_log:
                wandb.log({
                    "iter": iter_num,
                    "train/loss": losses["train"],
                    "val/loss": losses["val"],
                    "lr": lr,
                    "mfu": running_mfu * 100,
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

        iter_num += 1
        local_iter_num += 1

        if iter_num > max_iters:
            break

    # ---- cleanup ----
    if ddp:
        destroy_process_group()

    final_losses = {"train_loss": float(losses["train"]) if "losses" in dir() else None,
                    "val_loss": float(best_val_loss),
                    "out_dir": out_dir}

    if wandb_log and master_process:
        wandb.finish()

    print(f"\nDone: {wandb_run_name}  best_val_loss={best_val_loss:.4f}\n")
    return final_losses



if __name__ == "__main__":
    # Parse CLI: first positional args are config files, --key=value are overrides
    # 1. Define the search space
    models_to_test = [
        "config/train_shakespeare_char.py",   # Vanilla
        "config/train_shkspr_delta.py",        # Delta
        "config/train_shkspr_delta_prod.py"   # Delta Prod
    ]
    learning_rates = [5e-4, 1e-3, 1.5e-3, 3e-3]

    config_files = []
    cli_overrides = {}
    for config_path in models_to_test:
        for lr in learning_rates:
            # Extract simple name (e.g., 'delta_prod') from path for the label
            base_name = config_path.split('_')[-1].replace('.py', '')
            if 'char' in config_path: base_name = 'vanilla'
            
            # DYNAMIC NAME: This will show up in WandB as "vanilla_lr1.5e-03"
            run_id = f"{base_name}_lr{lr:.1e}"
            
            cli_overrides = {
                    "learning_rate": lr,
                    "wandb_run_name": run_id,
                    "out_dir": f"out-{run_id}" # Keeps checkpoints separate
                }
            for arg in sys.argv[1:]:
                if "=" not in arg:
                    assert not arg.startswith("--"), f"Unknown flag: {arg}"
                    config_files.append(arg)
                else:
                    assert arg.startswith("--")
                    key, val = arg.split("=", 1)
                    key = key[2:]
                    try:
                        val = literal_eval(val)
                    except (SyntaxError, ValueError):
                        pass
                    cli_overrides[key] = val

            if not config_files:
                # No config file given — just run with defaults + CLI overrides
                train(overrides=cli_overrides)
            else:
                for cf in config_files:
                    # Extract simple name (e.g., 'delta_prod') from path for the label
                    base_name = cf.split('_')[-1].replace('.py', '')
                    if 'char' in cf: wandb_run_name = 'vanilla'
                    
                    print(f"\nSTARTING RUN: {run_id}")
                    
                    # Pass the overrides to the train function
                    # print(f"\n>>> Running config: {cf}")
                    train(config_file=cf, overrides=cli_overrides)
