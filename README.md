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

Porting guide
-------------
Here are some quick learnings on how to port models so that they work as TorchScript.

**Expected integer literal for index:**

```python
    x_org = x                                                           
    for i in range(self.n_layers):
       x = self.conv_layers[i](x * x_mask)
           ~~~~~~~~~~~~~~~~~~ <--- HERE    
```

Seems like we can't use non-constant indices, per [this ticket](https://github.com/pytorch/pytorch/issues/39165) and [this one](https://github.com/pytorch/pytorch/issues/16123).

It's suggested to turn the loop into an `enumerate`. They also suggest [this](https://github.com/pytorch/pytorch/issues/16123#issuecomment-459424166).

For iteration over two lists using the same index, use `zip`. This,

```python
for i in range(self.n_layers):
  x = self.conv_layers[i](x * x_mask)
  x = self.norm_layers[i](x)
  x = self.relu_drop(x)
```

Becomes this,

```python
for conv_layer, norm_layer in zip(self.conv_layers, self.norm_layers):
  x = conv_layer(x * x_mask)
  x = norm_layer(x)
  x = self.relu_drop(x)
```

**lhs of assignment must be a variable, subscript, or starred expression:**

```python
    x, self.attn = self.attention(q, k, v, mask=attn_mask)
       ~~~~~~~~~ <--- HERE
```

There is an [open issue](https://github.com/pytorch/pytorch/issues/28303) for this.
For now it looks like the best way to solve this is to just break the assignment up.

Fix,

```python
    results = self.attention(q, k, v, mask=attn_mask)
    x = results[0]
    self.attn = results[1]
```

**torch.jit.frontend.NotSupportedError: multiple comprehension generators not supported yet:**

```python
  pad_shape = [item for sublist in l for item in sublist]
                <--- HERE
```

The above code is flattening a list of lists,

```
[[1, 2, 3], [4, 5, 6], [7, 8, 9]] 
> >>> pad_shape = [item for sublist in l for item in sublist] 
> >>> pad_shape                                     
[1, 2, 3, 4, 5, 6, 7, 8, 9]     
```

Unroll the double list comprehension. In this particular case, `torch.flatten()` [may work](https://stackoverflow.com/questions/55546873/how-do-i-flatten-a-tensor-in-pytorch/55546907),

```python
  pad_shape = l.flatten()
```

*Note: I'm not certain this is correct yet.*

Expected a value of type 'Tensor' for argument 'length' but instead found type 'int'.**

It may necessary to type hint argument signatures. 

```
_get_relative_embeddings(__torch__.attentions.MultiHeadAttention self, Tensor relative_embeddings, Tensor length) -> (Tensor):
```

Example:

```
def _get_relative_embeddings(self, relative_embeddings, length: int):
```

Or,

```
from typing import List
...

def convert_pad_shape(pad_shape: List[List[int]]):
```


Model-specific details
----------------------

**glow-tts notes**

Only two changes!

`FlowGenerator.forward` (the root module) needs the following changes:

Hardcode a few constructor args. These come from `inference.py`'s arg set:

```python
def forward(self, x, x_lengths, y=None, y_lengths=None, g=None, gen=False, noise_scale=1., length_scale=1.):
    y = None
    y_lengths = None
    noise_scale = .667
    length_scale = 1.0 
    gen = True
```

Return only the mel output at the end of the `forward` branch used in tracing:

```python
#return (y, y_m, y_logs, logdet), attn, logw, logw_, x_m, x_logs 
return y
```

