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

import math
import warnings
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from peft.tuners.tuners_utils import BaseTunerLayer
from peft.utils.other import transpose



def project(R, eps):
    I = torch.zeros((R.size(0), R.size(0)), dtype=R.dtype, device=R.device)
    diff = R - I
    norm_diff = torch.norm(diff)
    if norm_diff <= eps:
        return R
    else:
        return I + eps * (diff / norm_diff)

def project_batch(R, eps=1e-5):
    # scaling factor for each of the smaller block matrix
    eps = eps * 1 / torch.sqrt(torch.tensor(R.shape[0]))
    I = torch.zeros((R.size(1), R.size(1)), device=R.device, dtype=R.dtype).unsqueeze(0).expand_as(R)
    diff = R - I
    norm_diff = torch.norm(R - I, dim=(1, 2), keepdim=True)
    mask = (norm_diff <= eps).bool()
    out = torch.where(mask, R, I + eps * (diff / norm_diff))
    return out



class OFTLayer(BaseTunerLayer):
    # List all names of layers that may contain adapter weights
    adapter_layer_names = ["oft_linear", "oft_R"]
    
    def __init__(self, base_layer: nn.Module, **kwargs) -> None:
        
        # Just initialize the empty variables
        
        self.oft_linear = nn.ModuleDict({})
        self.oft_R = nn.ParameterDict({})
        self.r = 4
        
        self.num_gpus = torch.cuda.device_count()
        self.device_ids = list(range(self.num_gpus))

        # Mark the weight as unmerged
        self._disable_adapters = False
        self.merged_adapters = []
        
        # self.in_features = in_features
        # self.out_features = out_features
        self.kwargs = kwargs
        
        base_layer = self.get_base_layer()
        if isinstance(base_layer, nn.Linear):
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif hasattr(base_layer, "in_features") and hasattr(base_layer, "out_features"):
            # QuantLinear
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif hasattr(base_layer, "infeatures") and hasattr(base_layer, "outfeatures"):
            # QuantLinear
            in_features, out_features = base_layer.infeatures, base_layer.outfeatures
        else:
            print(f"{type(base_layer)}: {vars(base_layer)}")
            print(f"{base_layer._modules['oft_linear']}")
            raise ValueError(f"Unsupported layer type {type(base_layer)}")

        self.in_features = in_features
        self.out_features = out_features

    @property
    def merged(self) -> bool:
        return bool(self.merged_adapters)


    def update_layer(self, adapter_name, bias=False, r=4, eps=1e-5, is_coft=True, block_share=False, init_oft_weights=True):
        
        self.r = r
        
        # Set oft training arguments
        if self.r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")
        
        # Actual trainable parameters
        if self.r > 0:
            self.oft_linear[adapter_name] = nn.Linear(self.in_features, self.out_features)
            if block_share:
            # Initialized as an identity matrix
                self.oft_R_shape = [in_features // self.r, in_features // self.r]
                self.oft_R[adapter_name] = nn.Parameter(torch.zeros(self.oft_R_shape[0], self.oft_R_shape[0]), requires_grad=True)

                self.eps = eps * self.oft_R_shape[0] * self.oft_R_shape[0]
            else:
                # Initialized as an identity matrix
                self.oft_R_shape = [self.r, self.in_features // self.r, self.in_features // self.r]
                R = torch.zeros(self.oft_R_shape[1], self.oft_R_shape[1])
                R = torch.stack([R] * self.r)
                self.oft_R[adapter_name] = nn.Parameter(R, requires_grad=True)
                self.eps = eps * self.oft_R_shape[1] * self.oft_R_shape[1]
        if init_oft_weights:
            self.reset_oft_parameters(adapter_name, eps, block_share)

        weight = getattr(self.oft_linear, "weight", None)
        if weight is not None:
            # the layer is already completely initialized, this is an update
            if weight.dtype.is_floating_point or weight.dtype.is_complex:
                self.to(weight.device, dtype=weight.dtype)
            else:
                self.to(weight.device)
        self.set_adapter(self.active_adapters)
        
    def reset_oft_parameters(self, adapter_name, eps, block_share):
        if adapter_name in self.oft_linear.keys():
            
            self.oft_linear[adapter_name] = nn.Linear(self.in_features, self.out_features)
            
            if block_share:
            # Initialized as an identity matrix
                self.oft_R_shape = [in_features // self.r, in_features // self.r]
                self.oft_R[adapter_name] = nn.Parameter(torch.zeros(self.oft_R_shape[0], self.oft_R_shape[0]), requires_grad=True)

                self.eps = eps * self.oft_R_shape[0] * self.oft_R_shape[0]
            else:
                # Initialized as an identity matrix
                self.oft_R_shape = [self.r, self.in_features // self.r, self.in_features // self.r]
                R = torch.zeros(self.oft_R_shape[1], self.oft_R_shape[1])
                R = torch.stack([R] * self.r)
                self.oft_R[adapter_name] = nn.Parameter(R, requires_grad=True)
                self.eps = eps * self.oft_R_shape[1] * self.oft_R_shape[1]
        
    def cayley(self, data):
        r, c = list(data.shape)
        # Ensure the input matrix is skew-symmetric
        skew = 0.5 * (data - data.t())
        I = torch.eye(r, device=data.device)
        
        # Perform the Cayley parametrization
        Q = torch.mm(I + skew, torch.inverse(I - skew))
        return Q
    
    def cayley_batch(self, data):
        b, r, c = data.shape
        # Ensure the input matrix is skew-symmetric
        skew = 0.5 * (data - data.transpose(1, 2))
        I = torch.eye(r, device=data.device).unsqueeze(0).expand(b, r, c)

        # Perform the Cayley parametrization
        Q = torch.bmm(I - skew, torch.inverse(I + skew))

        return Q
    
    def block_diagonal(self, R):
        if self.block_share:
            # Create a list of R repeated block_count times
            blocks = [R] * self.r
        else:
            # Create a list of R slices along the third dimension
            blocks = [R[i, ...] for i in range(self.r)]

        # Use torch.block_diag to create the block diagonal matrix
        A = torch.block_diag(*blocks)

        return A


# Below code is based on https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
# and modified to work with PyTorch FSDP


#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------



class Linear(nn.Linear, OFTLayer):
    # OFT implemented in a dense layer
    def __init__(
        self,
        adapter_name: str,
        in_features, out_features, bias=False, r=4, eps=1e-5, is_coft=True, block_share=False,
        **kwargs,
    ) -> None:
        
        super(nn.Linear, self).__init__()
        OFTLayer.__init__(self, in_features=in_features, out_features=out_features)

        self.update_layer(adapter_name, bias=False, r=4, eps=1e-5, is_coft=True, block_share=False, init_oft_weights=True)

        self.set_adapter(adapter_name)

    def merge(self, safe_merge: bool = False) -> None:
        """
        Merge the active adapter weights into the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
        """
        if self.merged:
            warnings.warn(
                f"Already following adapters were merged {','.join(self.merged_adapters)}. "
                f"You are now additionally merging {','.join(self.active_adapters)}."
            )
        for active_adapter in self.active_adapters:
            if active_adapter in self.oft_linear.keys():
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    # orig_weights += self.get_delta_weight(active_adapter)
                    
                    block_diag = self.get_oft_diag(active_adapter)
                    fix_filt = self.oft_linear.weight.data.clone()                    

                    if not torch.isfinite(orig_weights).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )

                    fix_filt = torch.transpose(fix_filt, 0, 1)
                    filt = torch.mm(block_diagonal_matrix, fix_filt.to(dtype))
                    filt = torch.transpose(filt, 0, 1)
                    
                    # self.weight.data = orig_weights
                    self.oft_linear.weight = nn.Parameter(
                        filt
                        .type(_child_module.oft_R.dtype)
                        .to(_child_module.oft_linear.weight.device)
                    )
                else:
                    # I don't technically need get_delta_weight since I'm unmerging differently
                    block_diag = self.get_oft_diag(active_adapter)
                    fix_filt = self.oft_linear.weight.data.clone()                    

                    fix_filt = torch.transpose(fix_filt, 0, 1)
                    filt = torch.mm(block_diagonal_matrix, fix_filt.to(dtype))
                    filt = torch.transpose(filt, 0, 1)
                    
                    # self.weight.data = orig_weights
                    self.oft_linear.weight = nn.Parameter(
                        filt
                        .type(_child_module.oft_R.dtype)
                        .to(_child_module.oft_linear.weight.device)
                    )
                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        # remove adapter
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.oft_linear.keys():
                self.weight.data = self.oft_linear.weight.data

    def get_oft_diag(self, adapter) -> torch.Tensor:
        dtype = self.oft_R.dtype

        if self.block_share:
            if self.is_coft:
                with torch.no_grad():
                    self.oft_R.copy_(project(self.oft_R, eps=self.eps))
            orth_rotate = self.cayley(self.oft_R)
        else:
            if self.is_coft:
                with torch.no_grad():
                    self.oft_R.copy_(project_batch(self.oft_R, eps=self.eps))
            orth_rotate = self.cayley_batch(self.oft_R)

        # Block-diagonal parametrization
        block_diagonal_matrix = self.block_diagonal(orth_rotate)

        return block_diagonal_matrix
    
    def forward(self, x):
        orig_dtype = x.dtype
        dtype = self.oft_R.dtype

        if self.block_share:
            if self.is_coft:
                with torch.no_grad():
                    self.oft_R.copy_(project(self.oft_R, eps=self.eps))
            orth_rotate = self.cayley(self.oft_R)
        else:
            if self.is_coft:
                with torch.no_grad():
                    self.oft_R.copy_(project_batch(self.oft_R, eps=self.eps))
            orth_rotate = self.cayley_batch(self.oft_R)

        # Block-diagonal parametrization
        block_diagonal_matrix = self.block_diagonal(orth_rotate)
        
        # fix filter
        # print(self.oft_linear)
        fix_filt = self.oft_linear.weight.data
        fix_filt = torch.transpose(fix_filt, 0, 1)
        filt = torch.mm(block_diagonal_matrix, fix_filt.to(dtype))
        filt = torch.transpose(filt, 0, 1)
 
        # Apply the trainable identity matrix
        bias_term = self.oft_linear.bias.data if self.oft_linear.bias is not None else None
        out = nn.functional.linear(input=x, weight=filt, bias=bias_term)

        return out #.to(orig_dtype)


    def is_orthogonal(self, R, eps=1e-5):
        with torch.no_grad():
            RtR = torch.matmul(R.t(), R)
            diff = torch.abs(RtR - torch.eye(R.shape[1], dtype=R.dtype, device=R.device))
            return torch.all(diff < eps)

    def is_identity_matrix(self, tensor):
        if not torch.is_tensor(tensor):
            raise TypeError("Input must be a PyTorch tensor.")
        if tensor.ndim != 2 or tensor.shape[0] != tensor.shape[1]:
            return False
        identity = torch.eye(tensor.shape[0], device=tensor.device)
        return torch.all(torch.eq(tensor, identity))