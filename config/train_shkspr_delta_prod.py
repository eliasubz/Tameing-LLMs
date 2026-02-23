# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such

model_type = 'delta_product'
out_dir = 'out-delta-prod'
eval_interval = 50 # keep frequent because we'll overfit
eval_iters = 10
log_interval = 50 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = True

wandb_log = True # override via command line if you like
wandb_project = 'shakespeare-gated-delta-comparison'
wandb_run_name = 'gpt-delta-product'

dataset = 'shakespeare_char'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256 # context of up to 256 previous characters

# baby GPT model — n_head=2, n_embd=228 to match vanilla's 10.65M params (0.22% diff)
n_layer = 6
n_head = 2
n_embd = 228
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