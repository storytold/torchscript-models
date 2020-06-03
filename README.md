jit-models
==========
This project exists to turn PyTorch models into Torch JIT, which can be run in
any language with libtorch bindings.

How to develop
--------------
The best way is to import the *inference* code and work backwards. 

Try to comment out all of the model so `forward` only returns the input, then begin
opening things up.

Scripting vs tracing
--------------------
Scripting transforms the code. It can contain branches.

Tracing evaluates the codepath with example data, fixing the branches taken.

The two techniques can be combined in the various nodes in the module tree.

Model-specific details
----------------------

**glow-tts notes**

Only two changes!

`FlowGenerator.forward` (the root module) needs the following changes:

Hardcode a few constructor args. These come from `inference.py`'s arg set:

```
def forward(self, x, x_lengths, y=None, y_lengths=None, g=None, gen=False, noise_scale=1., length_scale=1.):
	y = None
	y_lengths = None
	noise_scale = .667
	length_scale = 1.0 
	gen = True
```

Return only the mel output at the end of the `forward` branch used in tracing:

```
#return (y, y_m, y_logs, logdet), attn, logw, logw_, x_m, x_logs 
return y
```

