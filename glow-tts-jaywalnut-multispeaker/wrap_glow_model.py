#!/usr/bin/env python3
# From: https://github.com/pytorch/pytorch/issues/20356#issuecomment-545572400
import torch

# Import glow-tts code
import sys
sys.path.append("./glow-tts-multispeaker")
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


#model_checkpoint_path = '/home/bt/models/glow-tts-jaywalnut/pretrained.pth'
model_checkpoint_path = '/home/bt/models/glow-tts-jaywalnut-multispeaker/G_998.pth'

hps, glow_model = load(model_checkpoint_path)
output_filename = 'glow_multispeaker.torchjit'


cmu_dict = cmudict.CMUDict(hps.data.cmudict_path)

tst_stn = "This is a test of the machine learning model. How does this sound?"
seq = text_to_sequence(tst_stn, ['english_cleaners'], cmu_dict)
sequence = np.array(seq)[None, :] # NB: None adds a new axis.
print("".join([symbols[c] for c in sequence[0]]))

torched = torch.from_numpy(sequence)

x_tst = torch.autograd.Variable(torched).long()
x_tst_lengths = torch.tensor([x_tst.shape[1]])


speaker_id = 1
speaker_g = torch.IntTensor([int(speaker_id)]).long()

print('\n tst_stn', len(tst_stn), tst_stn)
print('\n seq', len(seq), seq)
print('\n sequence', len(sequence), sequence.size, sequence)
print('\n torched', len(torched), torched.shape, torched)

print('\n x_tst', len(x_tst), x_tst)
print('\n x_tst_lengths', len(x_tst_lengths), x_tst_lengths)

print('\n speaker_g', len(speaker_g), speaker_g)


# NB: Tracing evaluates the model on input and unrolls and hardcodes branching
# and loops. Scripting allows these to remain by converting the entire program
# to TorchScript. Scripting seems MUCH faster, is also harder get working.
#trace_model = True # TODO: Try True.
if trace_model:
    print('Tracing model and saving as: {}'.format(output_filename))
    #mel_file = '/home/bt/dev/voder/data/mels/LJ002-0320.mel'
    #mel = torch.load(mel_file, map_location=torch.device('cpu'))
    # NB: Getting `Tracing failed sanity checks! Graphs differed across invocations!`
    inputs = (x_tst, x_tst_lengths, speaker_g)
    #traced_script_module = torch.jit.trace(glow_model, inputs, check_trace=True)
    traced_script_module = torch.jit.trace(glow_model, inputs, check_trace=False)
    traced_script_module.save(output_filename)
else:
    print('Saving script model as: {}'.format(output_filename))
    container = torch.jit.script(glow_model)
    container.save(output_filename)


