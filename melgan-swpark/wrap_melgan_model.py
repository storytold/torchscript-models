#!/usr/bin/env python3
# From: https://github.com/pytorch/pytorch/issues/20356#issuecomment-545572400
import torch

from collections import OrderedDict
import re
import yaml

from model_melgan.generator import load_melgan

#torch.set_default_tensor_type('torch.DoubleTensor')

#model_checkpoint_path = '/home/bt/models/melgan-swpark/trumptxlearn183_fa14347_7675.pt'
model_checkpoint_path = '/home/bt/models/melgan-swpark/ljs-high-pitch_8741385_6975.pt'
model_checkpoint_path = '/home/bt/models/melgan-swpark/trump-txlearn-2020.05.05_8741385_7525.pt'
model_checkpoint_path = '/home/bt/models/melgan-swpark/trump-txlearn-2020.05.05_8741385_13675.pt'
model_checkpoint_path = '/home/bt/models/melgan-swpark/melgan_trump_txlearn_2020-05-30_steno_chkpt9800.pt'

melgan_model = load_melgan(model_checkpoint_path)

#output_filename = 'shared_melgan_container2.pt'
#output_filename = 'melgan_ljs-high-pitch_container.pt'
output_filename = 'melgan_trump-txlearn-2020.05.05.jit'
output_filename = 'melgan_trump-txlearn-2020.05.05_13675.jit'
output_filename = 'melgan_trump_txlearn_2020-05-30_steno_chkpt9800.torchjit'

# NB: Tracing evaluates the model on input and unrolls and hardcodes branching
# and loops. Scripting allows these to remain by converting the entire program
# to TorchScript. Scripting seems MUCH faster, is also harder get working.
trace_model = True
if trace_model:
    print('Tracing model and saving as: {}'.format(output_filename))
    mel_file = '/home/bt/dev/voder/data/mels/LJ002-0320.mel'
    mel = torch.load(mel_file, map_location=torch.device('cpu'))
    # NB: Getting `Tracing failed sanity checks! Graphs differed across invocations!`
    traced_script_module = torch.jit.trace(melgan_model, mel, check_trace=True)
    traced_script_module.save(output_filename)
else:
    print('Saving script model as: {}'.format(output_filename))
    container = torch.jit.script(melgan_model)
    container.save(output_filename)


