# coding=utf-8
# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings
from typing import Any, List, Optional, Set, Tuple

import bitsandbytes as bnb
import torch
import torch.nn as nn
import torch.nn.functional as F

from peft.import_utils import is_bnb_4bit_available, is_bnb_available
from peft.utils.other import transpose

from .layer import OFTLayer

if is_bnb_available():

    class Linear8bitLt(OFTLayer):
        # Lora implemented in a dense layer
        def __init__(
            self,
            base_layer: nn.Module,
            adapter_name: str = "default",
            r: int = 0,
            alpha: float = 0.0,
            module_dropout: float = 0.0,
            init_weights: bool = True,
            **kwargs,
        ):
            super().__init__(base_layer)

            # Create adapter and set it active
            self._active_adapter = adapter_name
            self.update_layer(adapter_name, r, alpha, module_dropout, init_weights, **kwargs)

if is_bnb_4bit_available():

    class Linear4bit(OFTLayer):
        # Lora implemented in a dense layer
        def __init__(
            self,
            base_layer: nn.Module,
            adapter_name: str = "default",
            r: int = 0,
            alpha: float = 0.0,
            module_dropout: float = 0.0,
            init_weights: bool = True,
            **kwargs,
        ):
            
            super().__init__(base_layer)
            self.base_layer = base_layer
            self.in_features = base_layer.in_features
            self.out_features = base_layer.out_features
            OFTLayer.__init__(self, base_layer)

            # Create adapter and set it active
            # self._active_adapter = adapter_name
            self.update_layer(adapter_name, r, alpha, module_dropout, init_weights, **kwargs)

        def _get_delta_activations(
            self, adapter_name: str, prev_weight: Optional[torch.Tensor], *args: Any, **kwargs: Any
        ) -> torch.Tensor:
            delta_weight = self.get_delta_weight(adapter_name)

            base_layer = self.get_base_layer()
            print(f"base_layer: ", base_layer)
            print(f"attributes: ", vars(base_layer))
            print(f"weight: ", base_layer._parameters['weight'].shape)

            base_weight = base_layer.weight if prev_weight is None else prev_weight
            base_weight = torch.transpose(base_weight, 0, 1)

            if base_weight.shape[0] != delta_weight.shape[1]:
                # when in channels is not divisible by r
                delta_weight = delta_weight[: base_weight.shape[0], : base_weight.shape[0]]

            requires_conversion = not torch.is_autocast_enabled()
            if requires_conversion:
                base_weight = base_weight.to(delta_weight.dtype)

            weight = torch.mm(delta_weight, base_weight)
            weight = torch.transpose(weight, 0, 1)
            return weight

        def _forward_from_weight(
            self, input: torch.Tensor, weight: torch.Tensor, *args: Any, **kwargs: Any
        ) -> torch.Tensor:
            base_layer = self.get_base_layer()
            base_bias = base_layer.bias.data if base_layer.bias is not None else None
            
            
            print("input: ", input.shape)
            print("weight: ", weight.shape)
            weight = weight.view(-1, input.shape[-1])

            return F.linear(input=input, weight=weight, bias=base_bias)

        def __repr__(self) -> str:
            rep = super().__repr__()
            return "oft." + rep
