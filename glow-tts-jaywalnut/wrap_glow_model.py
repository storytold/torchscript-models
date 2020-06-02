#!/usr/bin/env python3
# From: https://github.com/pytorch/pytorch/issues/20356#issuecomment-545572400
import torch

# Import glow-tts code
import sys
sys.path.append("./glow-tts")
import utils
import models
import commons
from text.symbols import symbols

# Nvidia Apex
#from apex import amp
#from apex.parallel import DistributedDataParallel as DDP

# Data 
import numpy as np
from text import text_to_sequence, cmudict

trace_model = False
if sys.argv[1] == 'true':
  trace_model = True
elif sys.argv[1] == 'false':
  trace_model = False
else:
  raise Exception('Scripting or tracing?')

def load(model_checkpoint_path):
  #dist.init_process_group(backend='nccl', init_method='env://', world_size=n_gpus, rank=rank)
  #torch.manual_seed(hps.train.seed)
  #torch.cuda.set_device(rank)

  #train_dataset = TextMelLoader(hps.data.training_files, hps.data)
  #train_sampler = torch.utils.data.distributed.DistributedSampler(
  #    train_dataset,
  #    num_replicas=n_gpus,
  #    rank=rank,
  #    shuffle=True)
  #collate_fn = TextMelCollate(1)
  #train_loader = DataLoader(train_dataset, num_workers=8, shuffle=False,
  #    batch_size=hps.train.batch_size, pin_memory=True,
  #    drop_last=True, collate_fn=collate_fn, sampler=train_sampler)
  #if rank == 0:
  #  val_dataset = TextMelLoader(hps.data.validation_files, hps.data)
  #  val_loader = DataLoader(val_dataset, num_workers=8, shuffle=False,
  #      batch_size=hps.train.batch_size, pin_memory=True,
  #      drop_last=True, collate_fn=collate_fn)

  """
  hps = utils.get_hparams()

  generator = models.FlowGenerator(
      n_vocab=len(symbols), 
      out_channels=hps.data.n_mel_channels, 
      **hps.model) #.cuda(rank)

  optimizer_g = commons.Adam(generator.parameters(), scheduler=hps.train.scheduler, dim_model=hps.model.hidden_channels, warmup_steps=hps.train.warmup_steps, lr=hps.train.learning_rate, betas=hps.train.betas, eps=hps.train.eps)

  #if hps.train.fp16_run:
  #  generator, optimizer_g._optim = amp.initialize(generator, optimizer_g._optim, opt_level="O1")

  generator = DDP(generator)
  _, _, _, epoch_str = utils.load_checkpoint(model_checkpoint_path, generator, optimizer_g)

  #optimizer_g.step_num = (epoch_str - 1) * len(train_loader)
  #optimizer_g._update_learning_rate()
  """
  model_dir = "./logs/base/"
  hps = utils.get_hparams_from_dir(model_dir)
  print('Checkpoint path: {}'.format(model_checkpoint_path))

  with torch.no_grad():
    # NB(bt): This should fix the following:
    # RuntimeError: Cannot insert a Tensor that requires grad as a constant. 
    # Consider making it a parameter or input, or detaching the gradient                                                                                                                   Tensor:                                                                                                                       
    model = models.FlowGenerator(
        len(symbols),
        out_channels=hps.data.n_mel_channels,
        **hps.model) #.to("cuda")

    utils.load_checkpoint(model_checkpoint_path, model)
    model.decoder.store_inverse() # do not calcuate jacobians for fast decoding
    _ = model.eval()

  return (hps, model)

"""
model_hparams = {
  "hidden_channels": 192,
  "filter_channels": 768,
  "filter_channels_dp": 256,
  "kernel_size": 3,
  "p_dropout": 0.1,
  "n_blocks_dec": 12,
  "n_layers_enc": 6,
  "n_heads": 2,
  "p_dropout_dec": 0.05,
  "dilation_rate": 1,
  "kernel_size_dec": 5,
  "n_block_layers": 4,
  "n_sqz": 2,
  "prenet": True,
  "mean_only": True,
  "hidden_channels_enc": 192,
  "hidden_channels_dec": 192,
  "window_size": 4
}

# TODO:
# num_symbols = len(symbols)
num_symbols = 62 # TODO: Fake
num_mel_channels = 80 # TODO: Verify

load(model_checkpoint_path)

generator = models.FlowGenerator(num_symbols,
    out_channels=num_mel_channels,
    **model_hparams) #.to("cuda")

scheduler = "noam"
hidden_channels = 192
warmup_steps = 4000
learning_rate = 1e0
betas = [0.9, 0.98]
eps = 1e-9

optimizer = commons.Adam(generator.parameters(), 
  scheduler=scheduler, 
  dim_model=hidden_channels, 
  warmup_steps=warmup_steps, 
  lr=learning_rate, 
  betas=betas, 
  eps=eps)

#if hps.train.fp16_run:
#  generator, optimizer_g._optim = amp.initialize(generator, optimizer_g._optim, opt_level="O1")
#""
#_, _, _, epoch_str = utils.load_checkpoint(
#  utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"), generator, optimizer_g)
#""

glow_model, optimizer, learning_rate, iteration = \
    utils.load_checkpoint(model_checkpoint_path, generator, optimizer)
"""

#model_checkpoint_path = '/home/bt/models/glow-tts-jaywalnut/pretrained.pth'
model_checkpoint_path = '/home/bt/models/glow-tts-jaywalnut/G_1007.pth'

hps, glow_model = load(model_checkpoint_path)
output_filename = 'output.torchjit'


cmu_dict = cmudict.CMUDict(hps.data.cmudict_path)

tst_stn = " testing this text to speech generator "
sequence = np.array(text_to_sequence(tst_stn, ['english_cleaners'], cmu_dict))[None, :]
print("".join([symbols[c] for c in sequence[0]]))

x_tst = torch.autograd.Variable(torch.from_numpy(sequence)).long()
x_tst_lengths = torch.tensor([x_tst.shape[1]])


# NB: Tracing evaluates the model on input and unrolls and hardcodes branching
# and loops. Scripting allows these to remain by converting the entire program
# to TorchScript. Scripting seems MUCH faster, is also harder get working.
#trace_model = True # TODO: Try True.
if trace_model:
    print('Tracing model and saving as: {}'.format(output_filename))
    #mel_file = '/home/bt/dev/voder/data/mels/LJ002-0320.mel'
    #mel = torch.load(mel_file, map_location=torch.device('cpu'))
    # NB: Getting `Tracing failed sanity checks! Graphs differed across invocations!`
    inputs = (x_tst, x_tst_lengths)
    #traced_script_module = torch.jit.trace(glow_model, inputs, check_trace=True)
    traced_script_module = torch.jit.trace(glow_model, inputs, check_trace=True)
    traced_script_module.save(output_filename)
else:
    print('Saving script model as: {}'.format(output_filename))
    container = torch.jit.script(glow_model)
    container.save(output_filename)


