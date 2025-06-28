import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from typing import Optional, List, Any


class DoRALayer():
    def __init__(
            self,
            r: int,
            lora_alpha: int,
            lora_dropout: float,
            merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights
        self._caches: dict[str, Any] = {}

    def _cache_store(self, key: str, value: Any) -> None:
        self._caches[key] = value

    def _cache_pop(self, key: str) -> Any:
        value = self._caches.pop(key)
        return value


class Linear(nn.Linear, DoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
            self,
            in_features: int,
            out_features: int,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.,
            fan_in_fan_out: bool = False,
            # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
            merge_weights: bool = True,
            **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        DoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)

        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
            lora_weight = self.lora_B @ self.lora_A
            weight_norm = self._get_weight_norm(self.weight, lora_weight, scaling=0.0)
            self.magnitude = nn.Parameter(weight_norm, requires_grad=True)

        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize B the same way as the default for nn.Linear and A to zero
            # this is different than what is described in the paper but should not affect performance
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def train(self, mode: bool = True):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                # unmerged
                if self.r > 0:
                    # self.weight.data -= T(self.lora_B @ self.lora_A) * self.scaling
                    weight = self.weight
                    delta_weight = T(self.lora_B @ self.lora_A) * self.scaling
                    weight_norm = self._cache_pop("weight_norm")
                    dora_factor = self.magnitude / weight_norm
                    weight_orig = weight.data / dora_factor.view(-1, 1) - delta_weight
                    self.weight.data = weight_orig
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0:
                    # self.weight.data += T(self.lora_B @ self.lora_A) * self.scaling
                    orig_weight = self.weight
                    delta_weight = T(self.lora_B @ self.lora_A) * self.scaling
                    weight_norm = self._get_weight_norm(orig_weight, delta_weight, scaling=1.0).detach()
                    self._cache_store('weight_norm', weight_norm)
                    dora_factor = self.magnitude / weight_norm
                    self.weight.data = dora_factor.view(-1, 1) * (orig_weight + delta_weight)

                self.merged = True

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        if self.r > 0 and not self.merged:
            result = F.linear(x, T(self.weight), bias=self.bias)
            # result += (self.lora_dropout(x) @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)) * self.scaling
            # return result
            result += self.apply_dora(x)
            return result
        else:
            return F.linear(x, T(self.weight), bias=self.bias)

    def _get_weight_norm(self, weight, lora_weight, scaling) -> torch.Tensor:
        # calculate L2 norm of weight matrix, column-wise
        weight = weight + scaling * lora_weight
        weight_norm = torch.linalg.norm(weight, dim=1)
        return weight_norm

    def apply_dora(self, x):
        x = self.lora_dropout(x)
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        lora_weight = self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)
        magnitude = self.magnitude
        weight_norm = self._get_weight_norm(self.weight, lora_weight, self.scaling)
        weight_norm = weight_norm.detach()
        mag_norm_scale = (magnitude / weight_norm).view(1, -1)
        lora_x = x @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)
        result_dora = (mag_norm_scale - 1) * (
            F.linear(x, T(self.weight), bias=self.bias)
        ) + mag_norm_scale * lora_x * self.scaling

        return result_dora

class MergedLinear(nn.Linear, DoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
            self,
            in_features: int,
            out_features: int,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.,
            enable_lora: List[bool] = [False],
            fan_in_fan_out: bool = False,
            merge_weights: bool = True,
            **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        DoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)
        assert out_features % len(enable_lora) == 0, \
            'The length of enable_lora must divide out_features'
        self.enable_lora = enable_lora
        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if r > 0 and any(enable_lora):
            self.lora_A = nn.Parameter(
                self.weight.new_zeros((r * sum(enable_lora), in_features)))
            self.lora_B = nn.Parameter(
                self.weight.new_zeros((out_features // len(enable_lora) * sum(enable_lora), r))
            )  # weights for Conv1D with groups=sum(enable_lora)
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
            # Compute the indices
            self.lora_ind = self.weight.new_zeros(
                (out_features,), dtype=torch.bool
            ).view(len(enable_lora), -1)
            self.lora_ind[enable_lora, :] = True
            self.lora_ind = self.lora_ind.view(-1)

            lora_weight = self.merge_AB()
            weight_norm = self._get_weight_norm(self.weight, lora_weight, scaling=0.0)
            self.magnitude = nn.Parameter(weight_norm[self.lora_ind], requires_grad=True)

        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)

    def init_dora(self):
        del self.magnitude
        lora_weight = self.merge_AB()
        weight_norm = self._get_weight_norm(self.weight, lora_weight, scaling=0.0)
        self.magnitude = nn.Parameter(weight_norm[self.lora_ind], requires_grad=True)

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def zero_pad(self, x):
        result = x.new_zeros((len(self.lora_ind), *x.shape[1:]))
        result[self.lora_ind] = x
        return result

    def orig_pad(self, x, orig):
        result = x.new_zeros((len(self.lora_ind), *x.shape[1:]))
        result[self.lora_ind] = x
        result[~self.lora_ind] = orig[~self.lora_ind]
        return result

    def merge_AB(self):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        delta_w = F.conv1d(
            self.lora_A.unsqueeze(0),
            self.lora_B.unsqueeze(-1),
            groups=sum(self.enable_lora)
        ).squeeze(0)
        return T(self.zero_pad(delta_w))

    def _get_weight_norm(self, weight, lora_weight, scaling) -> torch.Tensor:
        # calculate L2 norm of weight matrix, column-wise
        weight = weight + scaling * lora_weight
        weight_norm = torch.linalg.norm(weight, dim=1)
        return weight_norm

    def apply_dora(self, x):
        x = self.lora_dropout(x)
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        lora_weight = self.merge_AB()

        weight_norm = self._get_weight_norm(self.weight, lora_weight, self.scaling)
        weight_norm = weight_norm.detach()

        magnitude = self.orig_pad(self.magnitude, weight_norm)

        mag_norm_scale = (magnitude / weight_norm).view(1, -1)
        lora_x = x @ T(lora_weight.T)
        result_dora = (mag_norm_scale - 1) * (
            F.linear(x, T(self.weight), bias=self.bias)
        ) + mag_norm_scale * lora_x * self.scaling

        return result_dora

    def train(self, mode: bool = True):
        nn.Linear.train(self, mode)
        if mode:
            if self.merge_weights and self.merged:
                # Make sure that the weights are not merged
                if self.r > 0 and any(self.enable_lora):
                    # self.weight.data -= self.merge_AB() * self.scaling
                    weight = self.weight
                    delta_weight = self.merge_AB() * self.scaling
                    weight_norm = self._cache_pop("weight_norm")
                    magnitude = self.orig_pad(self.magnitude, weight_norm)

                    dora_factor = magnitude / weight_norm
                    weight_orig = weight.data / dora_factor.view(-1, 1) - delta_weight
                    self.weight.data = weight_orig

                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # Merge the weights and mark it
                if self.r > 0 and any(self.enable_lora):
                    # self.weight.data += self.merge_AB() * self.scaling
                    orig_weight = self.weight
                    delta_weight = self.merge_AB() * self.scaling
                    weight_norm = self._get_weight_norm(orig_weight, delta_weight, scaling=1.0).detach()
                    self._cache_store("weight_norm", weight_norm)
                    dora_factor = self.orig_pad(self.magnitude, weight_norm) / weight_norm

                    self.weight.data = dora_factor.view(-1, 1) * (orig_weight + delta_weight)

                self.merged = True

    def forward(self, x: torch.Tensor):
        def T(w):
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        if self.merged:
            return F.linear(x, T(self.weight), bias=self.bias)
        else:
            result = F.linear(x, T(self.weight), bias=self.bias)
            if self.r > 0:
                result += self.apply_dora(x)
            return result


import torch
import torch.nn as nn

from typing import Dict


def mark_only_lora_as_trainable(model: nn.Module, bias: str = 'none') -> None:
    for n, p in model.named_parameters():
        if 'lora_' not in n:
            p.requires_grad = False
    if bias == 'none':
        return
    elif bias == 'all':
        for n, p in model.named_parameters():
            if 'bias' in n:
                p.requires_grad = True
    elif bias == 'lora_only':
        for m in model.modules():
            if isinstance(m, DoRALayer) and \
                hasattr(m, 'bias') and \
                m.bias is not None:
                    m.bias.requires_grad = True
    else:
        raise NotImplementedError


def lora_state_dict(model: nn.Module, bias: str = 'none') -> Dict[str, torch.Tensor]:
    my_state_dict = model.state_dict()
    if bias == 'none':
        return {k: my_state_dict[k] for k in my_state_dict if 'lora_' in k}
    elif bias == 'all':
        return {k: my_state_dict[k] for k in my_state_dict if 'lora_' in k or 'bias' in k}
    elif bias == 'dora_only':
        to_return = {}
        for k in my_state_dict:
            if 'lora_' in k:
                to_return[k] = my_state_dict[k]
                bias_name = k.split('lora_')[0]+'bias'
                if bias_name in my_state_dict:
                    to_return[bias_name] = my_state_dict[bias_name]
        return to_return
    else:
        raise NotImplementedError


