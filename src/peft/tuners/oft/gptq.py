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

import torch

from peft.tuners.oft.layer import OFTLayer


class QuantLinear(torch.nn.Module, OFTLayer):
    def __init__(
        self,
        adapter_name,
        quant_linear_module,
        bias: bool = False, 
        r: int = 4, 
        eps: float = 1e-5, 
        is_coft: bool = True, 
        block_share: bool = False, 
        init_oft_weights: bool = True,
        **kwargs,
    ):
        torch.nn.Module.__init__(self)
        LoraLayer.__init__(
            self, in_features=quant_linear_module.infeatures, out_features=quant_linear_module.outfeatures
        )
        self.quant_linear_module = quant_linear_module
        self.weight = quant_linear_module.qweight
        init_lora_weights = kwargs.pop("init_oft_weights", True)
        self.update_layer(adapter_name, bias, r, eps, is_coft, block_share, init_oft_weights)
        self.set_adapter(adapter_name)

    def forward(self, x: torch.Tensor):
        # note: logic differs from default Linear because merging is not supported
        result = self.quant_linear_module(x)

        if self.disable_adapters:
            return result

        for active_adapter in self.active_adapters:
            if active_adapter not in self.oft_linear.keys():
                continue
            oft_linear = self.oft_linear[active_adapter]
            oft_R = self.oft_R[active_adapter]

            requires_conversion = not torch.is_autocast_enabled()
            if requires_conversion:
                expected_dtype = result.dtype
                x = x.to(oft_linear.weight.dtype)
                
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
            output = nn.functional.linear(input=x, weight=filt, bias=bias_term)

            if requires_conversion:
                output = output.to(expected_dtype)
            result = output    
            
        return result

    # TODO: Check if it is better as suggested by users https://github.com/PanQiWei/AutoGPTQ/pull/102
    # def reset_lora_parameters(self, adapter_name):
    #     if adapter_name in self.lora_A.keys():
    #         torch.nn.init.xavier_uniform_(self.lora_A[adapter_name].weight)
    #         torch.nn.init.zeros_(self.lora_B[adapter_name].weight)
