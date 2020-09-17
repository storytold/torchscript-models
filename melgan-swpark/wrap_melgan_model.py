#!/usr/bin/env python3
# From: https://github.com/pytorch/pytorch/issues/20356#issuecomment-545572400
import torch

from collections import OrderedDict
import argparse
import re
import os
import sys
import yaml

from model_melgan.generator import load_melgan

#torch.set_default_tensor_type('torch.DoubleTensor')

DEFAULT_NAME = 'melgan-vocodes.torchjit'

def parse_args():
  parser = argparse.ArgumentParser('JITswap the melgan')
  parser.add_argument('--model_file', type=str, required=True) # Input
  parser.add_argument('--output_file', type=str, required=False) # Explicitly set the filename
  parser.add_argument('--copy_name', default=False, action='store_true') # Save as '{model_name}.torchjit'
  parser.add_argument('--output_directory', type=str, required=False, default='./OUTPUT/')

  args = parser.parse_args()

  if not args.model_file.endswith('.pt'):
    raise Exception('Melgan model filename `{}` does not end with extention `.pt`.'.format(args.model_file))

  if args.output_file and not args.output_file.endswith('.torchjit'):
    raise Exception('(Optional) Melgan JIT output filename `{}` does not end with extention `.torchjit`.'.format(args.output_file))

  if args.copy_name:
    if args.output_file:
      raise Exception('Cannot --copy_name and set an explicit --output_file')
    basename = os.path.basename(args.model_file)
    args.output_file = os.path.splitext(basename)[0] + '.torchjit'
  else:
    args.output_file = DEFAULT_NAME

  return args

args = parse_args()


model_checkpoint_path = args.model_file
output_filename = os.path.join(args.output_directory, args.output_file)

print('model file', model_checkpoint_path)
print('output file', output_filename)

melgan_model = load_melgan(model_checkpoint_path)


# NB: Tracing evaluates the model on input and unrolls and hardcodes branching
# and loops. Scripting allows these to remain by converting the entire program
# to TorchScript. Scripting seems MUCH faster, is also harder get working.
trace_model = True
if trace_model:
    print('Tracing model and saving as: {}'.format(output_filename))
    mel_file = './sample/LJ002-0320.mel'
    mel = torch.load(mel_file, map_location=torch.device('cpu'))
    # NB: Getting `Tracing failed sanity checks! Graphs differed across invocations!`
    traced_script_module = torch.jit.trace(melgan_model, mel, check_trace=True)
    traced_script_module.save(output_filename)
else:
    print('Saving script model as: {}'.format(output_filename))
    container = torch.jit.script(melgan_model)
    container.save(output_filename)


