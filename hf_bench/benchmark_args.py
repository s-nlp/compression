# coding=utf-8
# Copyright 2018 The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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

from dataclasses import dataclass, field
from typing import Tuple

from transformers.utils import cached_property, is_torch_available, is_torch_tpu_available, logging, torch_required
from .benchmark_args_utils import BenchmarkArguments


if is_torch_available():
    import torch

if is_torch_tpu_available(check_device=False):
    import torch_xla.core.xla_model as xm


logger = logging.get_logger(__name__)

#TO MAKE IT WORK RIGHT
#https://github.com/stas00/ipyexperiments/blob/3db0bbac2e2e6f1873b105953d9a7b3b7ca491b1/ipyexperiments/utils/mem.py
import os
def get_nvml_gpu_id(torch_gpu_id):
    """
    Remap torch device id to nvml device id, respecting CUDA_VISIBLE_DEVICES. 
    If the latter isn't set return the same id
    """
    # if CUDA_VISIBLE_DEVICES is used automagically remap the id since pynvml ignores this env var
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        ids = list(map(int, os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")))
        return ids[torch_gpu_id] # remap
    else:
        return torch_gpu_id

@dataclass
class PyTorchBenchmarkArguments(BenchmarkArguments):

    deprecated_args = [
        "no_inference",
        "no_cuda",
        "no_tpu",
        "no_speed",
        "no_memory",
        "no_env_print",
        "no_multi_process",
    ]

    def __init__(self, **kwargs):
        """
        This __init__ is there for legacy code. When removing deprecated args completely, the class can simply be
        deleted
        """
        for deprecated_arg in self.deprecated_args:
            if deprecated_arg in kwargs:
                positive_arg = deprecated_arg[3:]
                setattr(self, positive_arg, not kwargs.pop(deprecated_arg))
                logger.warning(
                    f"{deprecated_arg} is depreciated. Please use --no_{positive_arg} or"
                    f" {positive_arg}={kwargs[positive_arg]}"
                )

        self.torchscript = kwargs.pop("torchscript", self.torchscript)
        self.torch_xla_tpu_print_metrics = kwargs.pop("torch_xla_tpu_print_metrics", self.torch_xla_tpu_print_metrics)
        self.fp16_opt_level = kwargs.pop("fp16_opt_level", self.fp16_opt_level)
        super().__init__(**kwargs)

    torchscript: bool = field(default=False, metadata={"help": "Trace the models using torchscript"})
    torch_xla_tpu_print_metrics: bool = field(default=False, metadata={"help": "Print Xla/PyTorch tpu metrics"})
    fp16_opt_level: str = field(
        default="O1",
        metadata={
            "help": (
                "For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']. "
                "See details at https://nvidia.github.io/apex/amp.html"
            )
        },
    )

    @cached_property
    @torch_required
    def _setup_devices(self) -> Tuple["torch.device", int]:
        logger.info("PyTorch: setting up devices")
        if not self.cuda:
            device = torch.device("cpu")
            n_gpu = 0
        elif is_torch_tpu_available():
            device = xm.xla_device()
            n_gpu = 0
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            n_gpu = torch.cuda.device_count()
        return device, n_gpu

    @property
    def is_tpu(self):
        return is_torch_tpu_available() and self.tpu

    @property
    @torch_required
    def device_idx(self) -> int:
        # TODO(PVP): currently only single GPU is supported
        torch_gpu_id = torch.cuda.current_device()
        nvml_gpu_id = get_nvml_gpu_id(torch_gpu_id)
        return nvml_gpu_id

    @property
    @torch_required
    def device(self) -> "torch.device":
        return self._setup_devices[0]

    @property
    @torch_required
    def n_gpu(self):
        return self._setup_devices[1]

    @property
    def is_gpu(self):
        return self.n_gpu > 0
