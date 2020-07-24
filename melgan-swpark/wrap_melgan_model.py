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
model_checkpoint_path = '/home/bt/models/melgan-swpark/melgan_multivoice-1.0_2020.06.15_12d5838_6275.pt'
model_checkpoint_path = '/home/bt/models/melgan-swpark/melgan_multivoice-1.0_2020.06.15_12d5838_6300.pt'
model_checkpoint_path = '/home/bt/models/melgan-swpark/melgan_multivoice-1.0_2020.06.15_12d5838_6475.pt'
model_checkpoint_path = '/home/bt/models/melgan-swpark/melgan_multivoice-1.0_2020.06.15_12d5838_6725.pt'
model_checkpoint_path = '/home/bt/models/melgan-swpark/melgan_vocodes1.0_multispeaker_2020_06_19_12d5838_6675.pt'
model_checkpoint_path = '/home/bt/models/melgan-swpark/melgan_vocodes1.0_multispeaker_2020_06_19_12d5838_6900.pt'
model_checkpoint_path = '/home/bt/models/melgan-swpark/melgan_vocodes1.0_multispeaker_2020_06_19_12d5838_7125.pt'
model_checkpoint_path = '/home/bt/models/melgan-swpark/melgan_vocodes1.0_multispeaker_2020_06_19_12d5838_7400.pt'
model_checkpoint_path = '/home/bt/models/melgan-swpark/vocodes1.0_12d5838_7900.pt'
model_checkpoint_path = '/home/bt/models/melgan-swpark/vocodes1.0_12d5838_9025.pt'
model_checkpoint_path = '/home/bt/models/melgan-swpark/steno1.5_12d5838_9225.pt' # ugh, forgot the naming scheme
model_checkpoint_path = '/home/bt/models/melgan-swpark/steno1.5_12d5838_9475.pt'
model_checkpoint_path = '/home/bt/models/melgan-swpark/steno1.5_12d5838_10275.pt'
model_checkpoint_path = '/home/bt/models/melgan-swpark/manyvoice1.0_12d5838_6300.pt'
model_checkpoint_path = '/home/bt/models/melgan-swpark/manyvoice1.0_12d5838_6410.pt'
model_checkpoint_path = '/home/bt/models/melgan-swpark/manyvoice1.0_12d5838_6590.pt'
model_checkpoint_path = '/home/bt/models/melgan-swpark/manyvoice1.0_2020-06-28_12d5838_6610.pt'
model_checkpoint_path = '/home/bt/models/melgan-swpark/manyvoice1.0_12d5838_6730.pt'
model_checkpoint_path = '/home/bt/models/melgan-swpark/manyvoice1.0_2020-06-28_12d5838_6730.pt'
model_checkpoint_path = '/home/bt/models/melgan-swpark/manyvoice1.0_2020-06-28_12d5838_7050.pt'
model_checkpoint_path = '/home/bt/models/melgan-swpark/manyvoice1.0_2020-06-28_12d5838_7940.pt'
model_checkpoint_path = '/home/bt/models/melgan-swpark/manyvoice1.0_2020-06-28_12d5838_8570.pt'
model_checkpoint_path = '/home/bt/models/melgan-swpark/manyvoice2.0_2020-07-08_12d5838_8640.pt'
model_checkpoint_path = '/home/bt/models/melgan-swpark/manyvoice2.0_2020-07-08_12d5838_8810.pt'
model_checkpoint_path = '/home/bt/models/melgan-swpark/manyvoice2.0_2020-07-08_12d5838_8880.pt'
model_checkpoint_path = '/home/bt/models/melgan-swpark/manyvoice2.0_2020-07-08_12d5838_8900.pt'
model_checkpoint_path = '/home/bt/models/melgan-swpark/manyvoice2.0_2020-07-08_12d5838_8940.pt'
model_checkpoint_path = '/home/bt/models/melgan-swpark/manyvoice2.0_2020-07-08_12d5838_9050.pt'
model_checkpoint_path = '/home/bt/models/melgan-swpark/manyvoice2.0_2020-07-08_12d5838_9330.pt'
model_checkpoint_path = '/home/bt/models/melgan-swpark/manyvoice3.0_2020-07-17_12d5838_9820.pt'
model_checkpoint_path = '/home/bt/models/melgan-swpark/manyvoice3.0_2020-07-17_12d5838_9860.pt'
model_checkpoint_path = '/home/bt/models/melgan-swpark/manyvoice3.0_2020-07-17_12d5838_9920.pt'
model_checkpoint_path = '/home/bt/models/melgan-swpark/manyvoice3.0_2020-07-17_12d5838_9950.pt'
model_checkpoint_path = '/home/bt/models/melgan-swpark/manyvoice3.0_2020-07-17_12d5838_9960.pt'
model_checkpoint_path = '/home/bt/models/melgan-swpark/manyvoice3.0_2020-07-17_12d5838_10050.pt'
model_checkpoint_path = '/home/bt/models/melgan-swpark/manyvoice3.0_2020-07-17_12d5838_10090.pt'

# renamed for posterity
#model_checkpoint_path = '/home/bt/models/melgan-swpark/melgan_vocodes1.0_multispeaker_2020.06.19_12d5838_experiment-2_chkpt-10400.pt'

melgan_model = load_melgan(model_checkpoint_path)

#output_filename = 'shared_melgan_container2.pt'
#output_filename = 'melgan_ljs-high-pitch_container.pt'
output_filename = 'melgan_trump-txlearn-2020.05.05.jit'
output_filename = 'melgan_trump-txlearn-2020.05.05_13675.jit'
output_filename = 'melgan_trump_txlearn_2020-05-30_steno_chkpt9800.torchjit'
output_filename = 'melgan_multivoice-1.0_2020.06.15_12d5838_6275.torchjit'
output_filename = 'melgan_multivoice-1.0_2020.06.15_12d5838_6300.torchjit'
output_filename = 'melgan_multivoice-1.0_2020.06.15_12d5838_6475.torchjit'
output_filename = 'melgan_multivoice-1.0_2020.06.15_12d5838_6725.torchjit'
output_filename = 'melgan_vocodes1.0_multispeaker_2020_06_19_12d5838_6675.torchjit'
output_filename = 'melgan_vocodes1.0_multispeaker_2020_06_19_12d5838_6900.torchjit'
output_filename = 'melgan_vocodes1.0_multispeaker_2020_06_19_12d5838_7125.torchjit'
output_filename = 'melgan_vocodes1.0_multispeaker_2020_06_19_12d5838_7400.torchjit'
output_filename = 'melgan-vocodes.torchjit' # Stop renaming them for a bit
output_filename = 'melgan_vocodes1.0_multispeaker_2020.06.19_12d5838_experiment-2_chkpt-10400.torchjit'
output_filename = 'OUTPUT/manyvoice1.0_2020-06-28_12d5838_6610.torchjit'
output_filename = 'OUTPUT/manyvoice1.0_2020-06-28_12d5838_6730.torchjit'
output_filename = 'OUTPUT/manyvoice1.0_2020-06-28_12d5838_7050.torchjit'
output_filename = 'OUTPUT/manyvoice2.0_2020-07-08_12d5838_8940.torchjit'
output_filename = 'OUTPUT/manyvoice2.0_2020-07-08_12d5838_9170.torchjit'
output_filename = 'OUTPUT/manyvoice2.0_2020-07-08_12d5838_9330.torchjit'
output_filename = 'OUTPUT/manyvoice3.0_2020-07-17_12d5838_9920.torchjit'
output_filename = 'OUTPUT/melgan-vocodes.torchjit' # Stop renaming them for a bit
output_filename = 'OUTPUT/manyvoice3.0_2020-07-17_12d5838_10090.torchjit'

#output_filename = 'OUTPUT/melgan-vocodes.torchjit' # Stop renaming them for a bit

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


