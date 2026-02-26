# train a miniature character-level shakespeare model
# using Native Sparse Attention (NSA) — combines compressed block attention,
# selected token attention, and sliding window attention with learned gating.

model_type = 'nsa'
out_dir = 'out-nsa'
eval_interval = 50 # keep frequent because we'll overfit
eval_iters = 10
log_interval = 50 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = True # override via command line if you like
wandb_project = 'transformers'
wandb_run_name = 'gpt-nsa'

dataset = 'shakespeare_char'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256 # context of up to 256 previous characters

# baby GPT model — same n_layer/n_embd as vanilla
# NSA's fla kernel requires n_head % (num_kv_heads * 16) == 0,
# so we use GQA: 16 query heads, 1 KV head  (head_dim = 384/16 = 24)
n_layer = 6
n_head = 16
n_embd = 384
num_kv_heads = 1
dropout = 0.2

learning_rate = 1e-3 # with baby networks can afford to go a bit higher
max_iters = 5000
lr_decay_iters = 5000 # make equal to max_iters usually
min_lr = 1e-4 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

warmup_iters = 100 # not super necessary potentially

# system
device = 'cuda'
compile = False # do not torch compile the model
