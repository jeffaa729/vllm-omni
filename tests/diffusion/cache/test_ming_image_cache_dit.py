# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Ming-Image Cache-DiT adapter coverage without model weights or CUDA."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from cache_dit import ForwardPattern

from vllm_omni.diffusion.cache.cachedit import CacheDiTBackend
from vllm_omni.diffusion.models.ming_image.transformer import MingImageTransformer2DModel

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_ming_image_cache_dit_wraps_only_main_layers():
    transformer = MingImageTransformer2DModel.__new__(MingImageTransformer2DModel)
    torch.nn.Module.__init__(transformer)
    transformer.layers = torch.nn.ModuleList([torch.nn.Identity()])
    pipeline = SimpleNamespace(transformer=transformer)

    with patch("vllm_omni.diffusion.cache.cachedit.backend.cache_dit.enable_cache") as enable_cache:
        backend = CacheDiTBackend()
        backend.enable(pipeline)

    (adapter,) = pipeline._cache_dit_targets
    assert adapter.transformer is transformer
    assert adapter.blocks == [transformer.layers]
    assert adapter.forward_pattern == [ForwardPattern.Pattern_3]
    # Both Ming variants use one transformer forward per denoise step, including
    # Design-Layer's batch-doubled classifier-free guidance.
    assert adapter.has_separate_cfg is False
    # The Z-Image block calls its first parameter `x`, not `hidden_states`.
    assert adapter.check_forward_pattern is False
    enable_cache.assert_called_once()
    assert enable_cache.call_args.args[0] is adapter
