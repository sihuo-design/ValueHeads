"""Head-level read / write access via forward pre-hooks on each layer's `o_proj`.

The input of `o_proj` has shape (batch, seq, num_heads * head_dim) and is the concatenation of the
per-head attention outputs, so head h of a layer occupies the slice [h * head_dim, (h + 1) * head_dim).
All hooks act on the last token position. Interventions (HeadMask, HeadSteer) skip the prompt forward
pass and edit every decoding step, i.e. the token fed in at each generation step (requires the KV cache,
the default, so that decoding steps have length 1).
"""
import torch

from .models import model_geometry, o_proj


class HeadHooks:
    """Base class: registers one pre-hook per layer; use as a context manager."""

    decode_only = False

    def __init__(self, model):
        self.model = model
        self.num_layers, self.num_heads, self.head_dim = model_geometry(model.config)
        self.handles = []

    def edit(self, layer, x):
        """Returns the (possibly modified) o_proj input `x` of `layer`."""
        raise NotImplementedError

    def _hook(self, layer):
        def pre_hook(module, args, kwargs):
            x = args[0] if args else kwargs["input"]
            if self.decode_only and x.shape[1] != 1:
                return None
            if args:
                return (self.edit(layer, args[0]),) + tuple(args[1:]), kwargs
            kwargs["input"] = self.edit(layer, kwargs["input"])
            return args, kwargs
        return pre_hook

    def __enter__(self):
        for layer in range(self.num_layers):
            self.handles.append(
                o_proj(self.model, layer).register_forward_pre_hook(self._hook(layer), with_kwargs=True)
            )
        return self

    def __exit__(self, *exc):
        for handle in self.handles:
            handle.remove()
        self.handles = []


class ActivationCollector(HeadHooks):
    """Records the last-token head activations of every layer for one forward pass."""

    def __init__(self, model):
        super().__init__(model)
        self.states = [None] * self.num_layers

    def edit(self, layer, x):
        self.states[layer] = x[0, -1].detach().float().cpu()
        return x

    @torch.no_grad()
    def collect(self, input_ids):
        """Returns a (num_layers, num_heads * head_dim) tensor for a single tokenized input."""
        self.model(input_ids=input_ids.to(self.model.device))
        return torch.stack(self.states)


class HeadMask(HeadHooks):
    """Ablates heads by scaling their output with a small factor `eps`.

    heads: {layer: [head, ...]}
    """

    decode_only = True

    def __init__(self, model, heads, eps=1e-4):
        super().__init__(model)
        self.heads = heads
        self.eps = eps

    def edit(self, layer, x):
        if layer not in self.heads:
            return x
        x = x.clone()
        d = self.head_dim
        for h in self.heads[layer]:
            x[:, -1, h * d:(h + 1) * d] *= self.eps
        return x


class HeadSteer(HeadHooks):
    """Shifts head activations along value directions: x <- x + alpha * sigma * dir.

    vectors: {layer: tensor(num_heads * head_dim)}, already scaled by sigma (zeros for untouched heads).
    """

    decode_only = True

    def __init__(self, model, vectors, alpha=1.0):
        super().__init__(model)
        self.vectors = vectors
        self.alpha = alpha

    def edit(self, layer, x):
        if layer not in self.vectors:
            return x
        x = x.clone()
        x[:, -1] += self.alpha * self.vectors[layer].to(x.device, x.dtype)
        return x
