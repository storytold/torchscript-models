#!/usr/bin/env python3
# From: https://github.com/pytorch/pytorch/issues/20356#issuecomment-545572400
import argparse
import os

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

DEFAULT_NAME = 'glow_tts_single_speaker.torchjit'

def parse_args():
  parser = argparse.ArgumentParser('Split audio into ingestible chunks')
  parser.add_argument('--model_file', type=str, required=True) # Input
  parser.add_argument('--output_file', type=str, required=False) # Explicitly set the filename
  parser.add_argument('--copy_name', default=False, action='store_true') # Save as '{model_name}.torchjit'
  parser.add_argument('--output_directory', type=str, required=False, default='./OUTPUT/')

  args = parser.parse_args()

  if not args.model_file.endswith('.pth'):
    raise Exception('Model filename `{}` does not end with extention `.pth`.'.format(args.model_file))

  if args.output_file and not args.output_file.endswith('.torchjit'):
    raise Exception('(Optional) JIT output filename `{}` does not end with extention `.torchjit`.'.format(args.output_file))

  if args.copy_name:
    if args.output_file:
      raise Exception('Cannot --copy_name and set an explicit --output_file')
    basename = os.path.basename(args.model_file)
    args.output_file = os.path.splitext(basename)[0] + '.torchjit'
  else:
    args.output_file = DEFAULT_NAME

  return args

args = parse_args()

trace_model = True # TODO: We're doing this for now.

torch.set_grad_enabled(False)

def load(model_checkpoint_path):
  model_dir = "./logs/base/"
  hps = utils.get_hparams_from_dir(model_dir)
  print('Checkpoint path: {}'.format(model_checkpoint_path))

  with torch.no_grad():
    # NB(bt): This should fix the following:
    # RuntimeError: Cannot insert a Tensor that requires grad as a constant.
    model = models.FlowGenerator(
        len(symbols),
        out_channels=hps.data.n_mel_channels,
        **hps.model) #.to("cuda")

    utils.load_checkpoint(model_checkpoint_path, model)
    model.decoder.store_inverse() # do not calcuate jacobians for fast decoding
    _ = model.eval()

  return (hps, model)

model_checkpoint_path = args.model_file
output_filename = os.path.join(args.output_directory, args.output_file)

hps, glow_model = load(model_checkpoint_path)

cmu_dict = cmudict.CMUDict(hps.data.cmudict_path)

tst_stn = "This is a test of the machine learning model. How does this sound? This is some longer text to make sure long inputs work. It should be good."
seq = text_to_sequence(tst_stn, ['english_cleaners'], cmu_dict)
sequence = np.array(seq)[None, :] # NB: None adds a new axis.

torched = torch.from_numpy(sequence)

x_tst = torch.autograd.Variable(torched).long()
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
    traced_script_module = torch.jit.trace(glow_model, inputs, check_trace=False)
    traced_script_module.save(output_filename)
else:
    print('Saving script model as: {}'.format(output_filename))
    container = torch.jit.script(glow_model)
    container.save(output_filename)

print('Input: {}'.format(model_checkpoint_path))
print('Output: {}'.format(output_filename))
print('Done.')

