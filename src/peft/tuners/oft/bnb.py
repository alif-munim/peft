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

import bitsandbytes as bnb
import torch
import torch.nn as nn
import torch.nn.functional as F

from peft.import_utils import is_bnb_4bit_available, is_bnb_available
from peft.utils.other import transpose

from .layer import OFTLayer, project, project_batch


if is_bnb_available():

    class Linear8bitLt(torch.nn.Module, OFTLayer):
        # Lora implemented in a dense layer
        def __init__(
            self,
            adapter_name,
            base_layer,
            bias: bool = False, 
            r: int = 4, 
            eps: float = 1e-5, 
            is_coft: bool = True, 
            block_share: bool = False, 
            init_oft_weights: bool = True,
            **kwargs,
        ) -> None:
            super().__init__()
            OFTLayer.__init__(self, in_features=base_layer.in_features, out_features=base_layer.out_features)
            self.base_layer = base_layer

            init_oft_weights = kwargs.pop("init_oft_weights", True)
            self.update_layer(adapter_name, bias, r, eps, is_coft, block_share, init_oft_weights)
            self.set_adapter(adapter_name)

        def merge(self, safe_merge: bool = False):
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
                if active_adapter not in self.oft_linear.keys():
                    continue
                warnings.warn(
                    "Merge lora module to 8-bit linear may get different generations due to rounding errors."
                )
                block_diagonal_matrix = self.get_oft_diag(active_adapter)

                weight = self.base_layer.weight
                state = self.base_layer.state
                if state.SCB is None:
                    state.SCB = weight.SCB

                # Dequantize the result of identity matrix and int8 weight because bitsandbytes does not support int8
                # dequantization directly
                im = torch.eye(weight.data.shape[-1]).contiguous().half().to(weight.device)
                im, imt, SCim, SCimt, coo_tensorim = bnb.functional.double_quant(im)
                im, Sim = bnb.functional.transform(im, "col32")
                if state.CxB is None:
                    state.CxB, state.SB = bnb.functional.transform(weight.data, to_order=state.formatB)
                out32, Sout32 = bnb.functional.igemmlt(im, state.CxB, Sim, state.SB)
                output = bnb.functional.mm_dequant(out32, Sout32, SCim, state.SCB, bias=None).t()

                # LoRA: original weights dequantized, to lora type + device, add lora data (B@A * alpha/scaling)
                # OFT: original weights dequantized, multiplied by orthogonal matrix 
                
                # fix_filt = self.oft_linear.weight.data
                fix_filt = output.to(block_diagonal_matrix.dtype).to(block_diagonal_matrix.device) 
                fix_filt = torch.transpose(fix_filt, 0, 1)
                filt = torch.mm(block_diagonal_matrix, fix_filt.to(block_diagonal_matrix.dtype))
                filt = torch.transpose(filt, 0, 1)
                
                # w_data = output.to(lora_data.dtype).to(lora_data.device) + lora_data
                w_data = filt
                
                if safe_merge and not torch.isfinite(w_data).all():
                    raise ValueError(
                        f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                    )

                self.base_layer.weight = bnb.nn.Int8Params(
                    w_data.to("cpu"), requires_grad=False, has_fp16_weights=weight.has_fp16_weights
                ).to(weight.device)
                state.reset_grads()
                self.merged_adapters.append(active_adapter)

        def unmerge(self):
            if not self.merged:
                warnings.warn("Already unmerged. Nothing to do.")
                return

            while len(self.merged_adapters) > 0:
                active_adapter = self.merged_adapters.pop()
                if active_adapter not in self.oft_linear.keys():
                    continue
                warnings.warn(
                    "Unmerge lora module to 8-bit linear may get different generations due to rounding errors."
                )
                block_diagonal_matrix = self.get_oft_diag(active_adapter)

                weight = self.base_layer.weight
                state = self.base_layer.state
                if state.SCB is None:
                    state.SCB = weight.SCB
                im = torch.eye(weight.data.shape[-1]).contiguous().half().to(weight.device)
                im, imt, SCim, SCimt, coo_tensorim = bnb.functional.double_quant(im)
                im, Sim = bnb.functional.transform(im, "col32")

                if state.CxB is None:
                    state.CxB, state.SB = bnb.functional.transform(weight.data, to_order=state.formatB)
                out32, Sout32 = bnb.functional.igemmlt(im, state.CxB, Sim, state.SB)
                output = bnb.functional.mm_dequant(out32, Sout32, SCim, state.SCB, bias=None).t()

                # w_data = output.to(lora_data.dtype).to(lora_data.device) - lora_data
                w_data = output.to(block_diagonal_matrix.dtype).to(block_diagonal_matrix.device) 
                
                self.base_layer.weight = bnb.nn.Int8Params(
                    w_data.to("cpu"), requires_grad=False, has_fp16_weights=weight.has_fp16_weights
                ).to(weight.device)
                state.reset_grads()

        def get_oft_diag(self, adapter):
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

        def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
            if self.disable_adapters:
                if self.merged:
                    self.unmerge()
                result = self.base_layer(x, *args, **kwargs)
            elif self.merged:
                result = self.base_layer(x, *args, **kwargs)
            else:
                result = self.base_layer(x, *args, **kwargs)
                for active_adapter in self.active_adapters:
                    if active_adapter not in self.oft_linear.keys():
                        continue
                    oft_linear = self.oft_linear[active_adapter]
                    oft_R = self.oft_R[active_adapter]

                    requires_conversion = not torch.is_autocast_enabled()
                    if requires_conversion:
                        expected_dtype = result.dtype
                        compute_dtype = oft_linear.weight.dtype
                        if x.dtype != compute_dtype:
                            x = x.to(compute_dtype)
                    
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


if is_bnb_4bit_available():

    class Linear4bit(torch.nn.Module, OFTLayer):
        # Lora implemented in a dense layer
        def __init__(
            self,
            base_layer: torch.nn.Module,
            adapter_name: str,
            bias: bool = False, 
            r: int = 4, 
            eps: float = 1e-5, 
            is_coft: bool = True, 
            block_share: bool = False, 
            init_oft_weights: bool = True,
            **kwargs,
        ) -> None:
            
            super().__init__()

            # print(base_layer)
            self.in_features = base_layer.in_features
            self.out_features = base_layer.out_features
            self.base_layer = base_layer
            
            self.block_share = block_share
            self.r = r
            self.bias = bias
            self.eps = eps
            self.is_coft = is_coft
            self.init_oft_weights = init_oft_weights
            
            OFTLayer.__init__(self, base_layer)
            self.update_layer(adapter_name, bias, r, eps, is_coft, block_share, init_oft_weights)
            
#             super().__init__()
#             OFTLayer.__init__(self, in_features=base_layer.in_features, out_features=base_layer.out_features)
#             self.base_layer = base_layer
            
#             self.is_coft = is_coft
#             self.block_share = block_share

#             init_oft_weights = kwargs.pop("init_oft_weights", True)
#             self.update_layer(adapter_name, bias, r, eps, is_coft, block_share, init_oft_weights)
#             self.set_adapter(adapter_name)

        def merge(self, safe_merge: bool = False):
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
                
            if adapter_names is None:
                adapter_names = self.active_adapters

            for active_adapter in adapter_names:
                if active_adapter not in self.oft_linear.keys():
                    continue
                warnings.warn(
                    "Merge lora module to 4-bit linear may get different generations due to rounding errors."
                )
                # Refer to https://gist.github.com/ChrisHayduk/1a53463331f52dca205e55982baf9930
                weight = self.base_layer.weight
                kwargs = weight.__dict__
                block_diagonal_matrix = self.get_oft_diag(active_adapter)
                
                # fix_filt = self.oft_linear.weight.data
                fix_filt = bnb.functional.dequantize_4bit(weight.data, weight.quant_state)
                fix_filt = torch.transpose(fix_filt, 0, 1)
                filt = torch.mm(block_diagonal_matrix, fix_filt.to(dtype))
                filt = torch.transpose(filt, 0, 1)

                # w_data = bnb.functional.dequantize_4bit(weight.data, weight.quant_state) + lora_data
                w_data = filt
                
                if safe_merge and not torch.isfinite(w_data).all():
                    raise ValueError(
                        f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                    )

                self.base_layer.weight = bnb.nn.Params4bit(w_data.to("cpu"), requires_grad=False, **kwargs).to(
                    weight.device
                )
                self.merged_adapters.append(active_adapter)

        def unmerge(self):
            if not self.merged:
                warnings.warn("Already unmerged. Nothing to do.")
                return

            while len(self.merged_adapters) > 0:
                active_adapter = self.merged_adapters.pop()
                if active_adapter not in self.oft_linear.keys():
                    continue
                warnings.warn(
                    "Unmerge lora module to 4-bit linear may get different generations due to rounding errors."
                )
                weight = self.base_layer.weight
                kwargs = weight.__dict__
                
                # block_diagonal_matrix = self.get_oft_diag(active_adapter)
                
                # fix_filt = self.oft_linear.weight.data
                # fix_filt = torch.transpose(fix_filt, 0, 1)
                # filt = torch.mm(block_diagonal_matrix, fix_filt.to(dtype))
                # filt = torch.transpose(filt, 0, 1)
                
                # w_data = bnb.functional.dequantize_4bit(weight.data, weight.quant_state) - lora_data
                w_data = bnb.functional.dequantize_4bit(weight.data, weight.quant_state) 
                self.base_layer.weight = bnb.nn.Params4bit(w_data.to("cpu"), requires_grad=False, **kwargs).to(
                    weight.device
                )

        def get_oft_diag(self, adapter):
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

        def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
            if self.disable_adapters:
                # print(f"Adapters disabled. Unmerging OFT layers.")
                if self.merged:
                    self.unmerge()
                result = self.base_layer.forward(x, *args, **kwargs)
            elif self.merged:
                # print(f"Adapters currently merged. Running forward pass.")
                result = self.base_layer.forward(x, *args, **kwargs)
            else:
                # print(f"Adapters not currently merged. Running forward pass.")
                result = self.base_layer.forward(x, *args, **kwargs)
                # As per Tim Dettmers, for 4bit, we need to defensively clone here.
                # The reason is that in some cases, an error can occur that backprop
                # does not work on a manipulated view. This issue may be solved with
                # newer PyTorch versions but this would need extensive testing to be
                # sure.
                result = result.clone()

                for active_adapter in self.active_adapters:
                    if active_adapter not in self.oft_linear.keys():
                        continue
                    oft_linear = self.oft_linear[active_adapter]
                    oft_R = self.oft_R[active_adapter]

                    requires_conversion = not torch.is_autocast_enabled()
                    if requires_conversion:
                        expected_dtype = result.dtype
                        compute_dtype = oft_linear.weight.dtype
                        if x.dtype != compute_dtype:
                            x = x.to(compute_dtype)
                    
                    orig_dtype = x.dtype
                    dtype = oft_R.dtype

                    if self.block_share:
                        if self.is_coft:
                            with torch.no_grad():
                                oft_R.copy_(project(oft_R, eps=self.eps))
                        orth_rotate = self.cayley(oft_R)
                    else:
                        if self.is_coft:
                            with torch.no_grad():
                                oft_R.copy_(project_batch(oft_R, eps=self.eps))
                        orth_rotate = self.cayley_batch(oft_R)

                    # Block-diagonal parametrization
                    block_diagonal_matrix = self.block_diagonal(orth_rotate)

                    # fix filter
                    # print(self.oft_linear)
                    fix_filt = oft_linear.weight.data
                    fix_filt = torch.transpose(fix_filt, 0, 1)
                    filt = torch.mm(block_diagonal_matrix, fix_filt.to(dtype))
                    filt = torch.transpose(filt, 0, 1)

                    # Apply the trainable identity matrix
                    bias_term = oft_linear.bias.data if oft_linear.bias is not None else None
                    output = nn.functional.linear(input=x, weight=filt, bias=bias_term)
                    
                    
                    if requires_conversion:
                        output = output.to(expected_dtype)
                    result = output

            return result
