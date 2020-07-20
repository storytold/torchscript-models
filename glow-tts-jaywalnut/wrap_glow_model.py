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


# John Oliver training started at G_845 (started at Jun22 23:44)
# Reportedly started sounding good at G_1189 (saved at Jun23 06:27)
# 344 epochs * 461 clips = 158,584 clip iterations (403 minutes of training)
# 344 epochs * 1968 seconds = 676,992 seconds iterated (403 minutes of training)
model_checkpoint_path = '/home/bt/models/glow-tts-jaywalnut/oliver/G_891.pth' # Oh god, that's awful
model_checkpoint_path = '/home/bt/models/glow-tts-jaywalnut/oliver/G_1168.pth'
model_checkpoint_path = '/home/bt/models/glow-tts-jaywalnut/oliver/G_1189.pth' # Much better, but lots of room for improvement
model_checkpoint_path = '/home/bt/models/glow-tts-jaywalnut/oliver/G_1652.pth' # Not much of an improvement (?)

# Queen started training at G_1110. (started at Jun23 16:41)
# Still sounds bad at G_1436. (saved at Jun24 00:27)
# 326 epochs * 88 clips = 28,688 clip iterations (646 minutes of training)
# 326 epochs * 372 seconds = 121,272 seconds (646 minutes of training)
model_checkpoint_path = '/home/bt/models/glow-tts-jaywalnut/queen/G_1185.pth' # Still sounds exactly like LJS
model_checkpoint_path = '/home/bt/models/glow-tts-jaywalnut/queen/G_1203.pth'


#model_checkpoint_path = '/home/bt/models/glow-tts-jaywalnut/samuel-l-jackson/glow-tts_samuel_l_jackson_ljstx_2020.07.05_expr-1_chkpt-2444.pth'

model_checkpoint_path = '/home/bt/models/glow-tts-jaywalnut/craig-ferguson/new/G_4682.pth'
model_checkpoint_path = '/home/bt/models/glow-tts-jaywalnut/craig-ferguson/old/G_2282.pth'


model_checkpoint_path = '/home/bt/models/glow-tts-jaywalnut/george-w-bush/G_1735.pth'
model_checkpoint_path = '/home/bt/models/glow-tts-jaywalnut/jimmy-carter/G_2391.pth'
model_checkpoint_path = '/home/bt/models/glow-tts-jaywalnut/jimmy-carter/glow-tts_jimmy_carter_ljstx_2020.07.05_expr-1_chkpt-2391.pth'
model_checkpoint_path = '/home/bt/models/glow-tts-jaywalnut/george-w-bush/G_1735.pth'
model_checkpoint_path = '/home/bt/models/glow-tts-jaywalnut/bill-nye/glow-tts_bill_nye_ljstx_2020.07.07_expr-1_chkpt-2510.pth'
model_checkpoint_path = '/home/bt/models/glow-tts-jaywalnut/judi-dench/G_4693.pth'
model_checkpoint_path = '/home/bt/models/glow-tts-jaywalnut/george-takei/G_4060.pth'
model_checkpoint_path = '/home/bt/models/glow-tts-jaywalnut/tom-hanks/G_3686.pth'
model_checkpoint_path = '/home/bt/models/glow-tts-jaywalnut/james-earl-jones/G_2573.pth'
model_checkpoint_path = '/home/bt/models/glow-tts-jaywalnut/samuel-l-jackson/2nd/G_4481.pth'
model_checkpoint_path = '/home/bt/models/glow-tts-jaywalnut/craig-ferguson/new/G_6204.pth'
model_checkpoint_path = '/home/bt/models/glow-tts-jaywalnut/david-cross/G_5029.pth'
model_checkpoint_path = '/home/bt/models/glow-tts-jaywalnut/samuel-l-jackson/2nd/G_5050.pth'
model_checkpoint_path = '/home/bt/models/glow-tts-jaywalnut/bryan-cranston/G_3606.pth'
model_checkpoint_path = '/home/bt/models/glow-tts-jaywalnut/craig-ferguson/new/G_7151.pth'
model_checkpoint_path = '/home/bt/models/glow-tts-jaywalnut/james-earl-jones/G_2573.pth'
model_checkpoint_path = '/home/bt/models/glow-tts-jaywalnut/samuel-l-jackson/2nd/G_5558.pth'
model_checkpoint_path = '/home/bt/models/glow-tts-jaywalnut/craig-ferguson/new/G_7657.pth'
model_checkpoint_path = '/home/bt/models/glow-tts-jaywalnut/hillary-clinton/G_2679.pth'
model_checkpoint_path = '/home/bt/models/glow-tts-jaywalnut/samuel-l-jackson/2nd/G_6808.pth'
model_checkpoint_path = '/home/bt/models/glow-tts-jaywalnut/david-cross/G_6791.pth'
model_checkpoint_path = '/home/bt/models/glow-tts-jaywalnut/hillary-clinton/2nd/G_5337.pth'
model_checkpoint_path = '/home/bt/models/glow-tts-jaywalnut/bryan-cranston/G_5916.pth'
model_checkpoint_path = '/home/bt/models/glow-tts-jaywalnut/david-cross/G_7425.pth'
model_checkpoint_path = '/home/bt/models/glow-tts-jaywalnut/samuel-l-jackson/2nd/G_7447.pth'
model_checkpoint_path = '/home/bt/models/glow-tts-jaywalnut/arnold-schwarzenegger/G_3812.pth'
model_checkpoint_path = '/home/bt/models/glow-tts-jaywalnut/tupac-shakur/2nd/G_3928.pth'
model_checkpoint_path = '/home/bt/models/glow-tts-jaywalnut/tupac-shakur/2nd/G_4925.pth'
model_checkpoint_path = '/home/bt/models/glow-tts-jaywalnut/arnold-schwarzenegger/G_5099.pth'
model_checkpoint_path = '/home/bt/models/glow-tts-jaywalnut/tupac-shakur/2nd/G_5128.pth'
model_checkpoint_path = '/home/bt/models/glow-tts-jaywalnut/tupac-shakur/2nd/G_5193.pth'
model_checkpoint_path = '/home/bt/models/glow-tts-jaywalnut/bryan-cranston/G_6838.pth'
model_checkpoint_path = '/home/bt/models/glow-tts-jaywalnut/hillary-clinton/2nd/G_6416.pth'
model_checkpoint_path = '/home/bt/models/glow-tts-jaywalnut/tupac-shakur/2nd/G_6341.pth'
model_checkpoint_path = '/home/bt/models/glow-tts-jaywalnut/tupac-shakur/glow-tts_tupac_shakur_ljstx_2020.07.16_expr-2_chkpt-6341.pth'
model_checkpoint_path = '/home/bt/models/glow-tts-jaywalnut/bryan-cranston/G_7792.pth'
model_checkpoint_path = '/home/bt/models/glow-tts-jaywalnut/craig-ferguson/new/G_7680.pth'
model_checkpoint_path = '/home/bt/models/glow-tts-jaywalnut/david-cross/G_7000.pth'
model_checkpoint_path = '/home/bt/models/glow-tts-jaywalnut/david-cross/G_4000.pth'
model_checkpoint_path = '/home/bt/models/glow-tts-jaywalnut/craig-ferguson/new/G_7680.pth'
model_checkpoint_path = '/home/bt/models/glow-tts-jaywalnut/neil-degrasse-tyson/glow-tts_neil_degrasse_tyson_ljstx_2020.07.05_expr-1_chkpt-2482.pth'
model_checkpoint_path = '/home/bt/models/glow-tts-jaywalnut/bryan-cranston/G_7795.pth'
model_checkpoint_path = '/home/bt/models/glow-tts-jaywalnut/hillary-clinton/2nd/G_7250.pth'
model_checkpoint_path = '/home/bt/models/glow-tts-jaywalnut/david-cross/G_9462.pth'
model_checkpoint_path = '/home/bt/models/glow-tts-jaywalnut/james-earl-jones/G_2573.pth'
model_checkpoint_path = '/home/bt/models/glow-tts-jaywalnut/samuel-l-jackson/2nd/G_9450.pth'
model_checkpoint_path = '/home/bt/models/glow-tts-jaywalnut/samuel-l-jackson/2nd/G_6000.pth'
model_checkpoint_path = '/home/bt/models/glow-tts-jaywalnut/samuel-l-jackson/2nd/G_4000.pth'
model_checkpoint_path = '/home/bt/models/glow-tts-jaywalnut/arnold-schwarzenegger/G_6955.pth'
model_checkpoint_path = '/home/bt/models/glow-tts-jaywalnut/peter-thiel/G_4252.pth'
model_checkpoint_path = '/home/bt/models/glow-tts-jaywalnut/john-oliver/2nd/G_4262.pth'
model_checkpoint_path = '/home/bt/models/glow-tts-jaywalnut/john-oliver/2nd/G_5046.pth'
model_checkpoint_path = '/home/bt/models/glow-tts-jaywalnut/peter-thiel/G_5072.pth'
model_checkpoint_path = '/home/bt/models/glow-tts-jaywalnut/arnold-schwarzenegger/G_8449.pth'
model_checkpoint_path = '/home/bt/models/glow-tts-jaywalnut/tupac-shakur/2nd/G_8238.pth'
model_checkpoint_path = '/home/bt/models/glow-tts-jaywalnut/larry-king/G_3783.pth' # Not yet...
model_checkpoint_path = '/home/bt/models/glow-tts-jaywalnut/j-k-simmons/G_3763.pth'
model_checkpoint_path = '/home/bt/models/glow-tts-jaywalnut/arnold-schwarzenegger/G_6955.pth'
model_checkpoint_path = '/home/bt/models/glow-tts-jaywalnut/arnold-schwarzenegger/G_8449.pth'
model_checkpoint_path = '/home/bt/models/glow-tts-jaywalnut/arnold-schwarzenegger/G_9024.pth'
model_checkpoint_path = '/home/bt/models/glow-tts-jaywalnut/tupac-shakur/2nd/G_8778.pth'
model_checkpoint_path = '/home/bt/models/glow-tts-jaywalnut/tom-hanks/G_3686.pth'
model_checkpoint_path = '/home/bt/models/glow-tts-jaywalnut/george-takei/G_4060.pth'
model_checkpoint_path = '/home/bt/models/glow-tts-jaywalnut/george-takei/G_2500.pth'
model_checkpoint_path = '/home/bt/models/glow-tts-jaywalnut/george-takei/G_3000.pth'
model_checkpoint_path = '/home/bt/models/glow-tts-jaywalnut/george-takei/G_2185.pth'
model_checkpoint_path = '/home/bt/models/glow-tts-jaywalnut/george-w-bush/G_1735.pth'
model_checkpoint_path = '/home/bt/models/glow-tts-jaywalnut/judi-dench/G_4693.pth'
model_checkpoint_path = '/home/bt/models/glow-tts-jaywalnut/judi-dench/G_3000.pth'
model_checkpoint_path = '/home/bt/models/glow-tts-jaywalnut/david-cross/G_9462.pth'
model_checkpoint_path = '/home/bt/models/glow-tts-jaywalnut/bryan-cranston/G_7795.pth'
model_checkpoint_path = '/home/bt/models/glow-tts-jaywalnut/neil-degrasse-tyson/glow-tts_neil_degrasse_tyson_ljstx_2020.07.05_expr-1_chkpt-2482.pth'
model_checkpoint_path = '/home/bt/models/glow-tts-jaywalnut/hillary-clinton/2nd/G_7250.pth'
model_checkpoint_path = '/home/bt/models/glow-tts-jaywalnut/james-earl-jones/G_2573.pth'
model_checkpoint_path = '/home/bt/models/glow-tts-jaywalnut/arnold-schwarzenegger/G_9045.pth'
model_checkpoint_path = '/home/bt/models/glow-tts-jaywalnut/craig-ferguson/new/G_7680.pth'
model_checkpoint_path = '/home/bt/models/glow-tts-jaywalnut/larry-king/G_4037.pth'
model_checkpoint_path = '/home/bt/models/glow-tts-jaywalnut/j-k-simmons/G_4017.pth'
model_checkpoint_path = '/home/bt/models/glow-tts-jaywalnut/john-oliver/2nd/G_5856.pth'
model_checkpoint_path = '/home/bt/models/glow-tts-jaywalnut/sam-altman/G_3654.pth' # Not quite
model_checkpoint_path = '/home/bt/models/glow-tts-jaywalnut/leonard-nimoy/G_3614.pth'

hps, glow_model = load(model_checkpoint_path)

#output_filename = 'glow-tts_solo_oliver_ljstx_2020.06.22_expr-1_chkpt-1189.torchjit'
#output_filename = 'glow-tts_solo_reagan_ljstx_2020.06.22_expr-1_chkpt-1084.torchjit'
#output_filename = 'glow-tts_solo_reagan_ljstx_2020.06.22_expr-1_chkpt-1406.torchjit'
output_filename = 'OUTPUT/craig_ferguson_new.torchjit'
output_filename = 'OUTPUT/glow-tts_neil_degrasse_tyson_ljstx_2020.07.05_expr-1_chkpt-2482.torchjit'
output_filename = 'OUTPUT/glow-tts_jimmy_carter_ljstx_2020.07.05_expr-1_chkpt-2391.torchjit'
output_filename = 'OUTPUT/glow-tts_bill_nye_ljstx_2020.07.07_expr-1_chkpt-2510.torchjit'
output_filename = 'OUTPUT/glow-tts_tupac_shakur_ljstx_2020.07.16_expr-2_chkpt-6341.torchjit'
output_filename = 'OUTPUT/arnold-a.torchjit'
output_filename = 'OUTPUT/arnold-b.torchjit'
output_filename = 'OUTPUT/arnold-c.torchjit'
output_filename = 'OUTPUT/glow_tts_single_speaker.torchjit'


cmu_dict = cmudict.CMUDict(hps.data.cmudict_path)

tst_stn = "This is a test of the machine learning model. How does this sound? This is some longer text to make sure long inputs work. It should be good."
seq = text_to_sequence(tst_stn, ['english_cleaners'], cmu_dict)
sequence = np.array(seq)[None, :] # NB: None adds a new axis.
#print("".join([symbols[c] for c in sequence[0]]))

torched = torch.from_numpy(sequence)

x_tst = torch.autograd.Variable(torched).long()
x_tst_lengths = torch.tensor([x_tst.shape[1]])

#print('\n tst_stn', len(tst_stn), tst_stn)
#print('\n seq', len(seq), seq)
#print('\n sequence', len(sequence), sequence.size, sequence)
#print('\n torched', len(torched), torched.shape, torched)

#print('\n x_tst', len(x_tst), x_tst)
#print('\n x_tst_lengths', len(x_tst_lengths), x_tst_lengths)

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

