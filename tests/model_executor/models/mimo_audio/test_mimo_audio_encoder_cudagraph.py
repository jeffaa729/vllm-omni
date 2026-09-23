# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from functools import lru_cache
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

pytestmark = pytest.mark.core_model


@lru_cache(maxsize=1)
def _mimo_llm_deps():
    from vllm_omni.model_executor.models.mimo_audio.mimo_audio_llm import (
        MIMO_INPUT_LOCAL_CUDAGRAPH_INPUT_KEY,
        MiMoAudioLLMForConditionalGeneration,
    )

    return (
        MIMO_INPUT_LOCAL_CUDAGRAPH_INPUT_KEY,
        MiMoAudioLLMForConditionalGeneration,
    )


class _InputLocalTransformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.call_count = 0

    def forward(
        self,
        *,
        inputs_embeds: torch.Tensor,
        return_dict: bool,
        is_causal: bool,
    ) -> SimpleNamespace:
        assert return_dict
        assert not is_causal
        self.call_count += 1
        return SimpleNamespace(last_hidden_state=inputs_embeds + 1)


def _model():
    _, model_cls = _mimo_llm_deps()
    model = object.__new__(model_cls)
    nn.Module.__init__(model)
    model.group_size = 2
    model.input_local_config = SimpleNamespace(hidden_size=3)
    model.input_local_transformer = _InputLocalTransformer()
    model.input_local_transformer_cudagraph_manager = None
    return model


@pytest.mark.cpu
def test_input_local_transformer_encoder_cudagraph_protocol() -> None:
    pytest.importorskip("vllm.v1.worker.encoder_cudagraph_defs")
    from vllm.model_executor.models.interfaces import supports_encoder_cudagraph

    input_key, _ = _mimo_llm_deps()
    model = _model()
    inputs = torch.arange(18, dtype=torch.float32).reshape(3, 2, 3)

    assert supports_encoder_cudagraph(model)

    config = model.get_encoder_cudagraph_config()
    assert config.modalities == []
    assert config.buffer_keys == [input_key]
    assert config.out_hidden_size == 3
    assert model.get_encoder_cudagraph_budget_range(
        SimpleNamespace(scheduler_config=SimpleNamespace(max_num_seqs=4))
    ) == (2, 8)

    specs = model.get_encoder_cudagraph_item_specs({input_key: inputs})
    assert [(spec.input_size, spec.output_tokens) for spec in specs] == [(6, 6)]

    selected = model.select_encoder_cudagraph_items({input_key: inputs}, [0])
    assert selected[input_key] is inputs
    empty = model.select_encoder_cudagraph_items({input_key: inputs}, [])
    assert empty[input_key].shape == (0, 2, 3)
    with pytest.raises(ValueError, match="aggregate MiMo item index"):
        model.select_encoder_cudagraph_items({input_key: inputs}, [1])


@pytest.mark.cpu
def test_input_local_transformer_capture_and_replay_buffers() -> None:
    pytest.importorskip("vllm.v1.worker.encoder_cudagraph_defs")
    input_key, _ = _mimo_llm_deps()
    model = _model()

    for num_groups in (1, 8, 32):
        capture = model.prepare_encoder_cudagraph_capture_inputs(
            token_budget=num_groups * model.group_size,
            max_batch_size=1,
            max_frames_per_batch=0,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        assert capture.values[input_key].shape == (num_groups, 2, 3)

    actual = torch.ones((1, 2, 3), dtype=torch.float32)
    replay = model.prepare_encoder_cudagraph_replay_buffers(
        {input_key: actual},
        max_batch_size=4,
        max_frames_per_batch=0,
    )
    assert replay.values[input_key] is actual

    for invalid_budget in (1, 3):
        with pytest.raises(ValueError, match="positive multiple"):
            model.prepare_encoder_cudagraph_capture_inputs(
                token_budget=invalid_budget,
                max_batch_size=1,
                max_frames_per_batch=0,
                device=torch.device("cpu"),
                dtype=torch.float32,
            )


@pytest.mark.cpu
def test_input_local_transformer_has_no_manager_on_cpu_and_stays_eager() -> None:
    model = _model()
    inputs = torch.zeros((2, 2, 3), dtype=torch.float32)

    assert model.input_local_transformer_cudagraph_manager is None
    output = model._run_generated_input_local_transformer(inputs)

    torch.testing.assert_close(output, inputs + 1)
    assert model.input_local_transformer.call_count == 1


@pytest.mark.cpu
def test_input_local_transformer_uses_attached_manager() -> None:
    input_key, _ = _mimo_llm_deps()
    model = _model()
    inputs = torch.arange(12, dtype=torch.float32).reshape(2, 2, 3)

    class _Manager:
        def __init__(self) -> None:
            self.received = None

        def is_captured(self) -> bool:
            return True

        def execute(self, mm_kwargs):
            self.received = mm_kwargs
            return [mm_kwargs[input_key].flatten(0, 1) + 2]

    manager = _Manager()
    model.set_input_local_transformer_cudagraph_manager(manager)

    output = model._run_generated_input_local_transformer(inputs)

    assert manager.received is not None
    assert manager.received[input_key] is inputs
    torch.testing.assert_close(output, inputs + 2)
    assert model.input_local_transformer.call_count == 0


@pytest.mark.cuda
def test_input_local_transformer_manager_matches_eager_across_budget_tiers() -> None:
    pytest.importorskip("vllm.v1.worker.encoder_cudagraph")
    from vllm.v1.worker.encoder_cudagraph import EncoderCudaGraphManager

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for encoder CUDA graph capture")

    input_key, _ = _mimo_llm_deps()
    model = _model().to(device="cuda")
    config = SimpleNamespace(
        compilation_config=SimpleNamespace(
            encoder_cudagraph_token_budgets=[2, 16, 64],
            encoder_cudagraph_max_vision_items_per_batch=1,
            encoder_cudagraph_max_frames_per_batch=0,
        ),
        model_config=SimpleNamespace(
            multimodal_config=SimpleNamespace(
                get_limit_per_prompt=lambda _modality: 0,
                mm_encoder_tp_mode="weights",
            )
        ),
        parallel_config=SimpleNamespace(tensor_parallel_size=1),
        scheduler_config=SimpleNamespace(max_num_seqs=32),
    )
    manager = EncoderCudaGraphManager(
        vllm_config=config,
        device=torch.device("cuda"),
        dtype=torch.float32,
        model=model,
    )
    manager.capture(graph_pool=torch.cuda.graph_pool_handle())
    model.set_input_local_transformer_cudagraph_manager(manager)

    for num_groups in (1, 8, 32):
        inputs = torch.arange(
            num_groups * 2 * 3,
            device="cuda",
            dtype=torch.float32,
        ).reshape(num_groups, 2, 3)
        expected = model.encoder_eager_forward({input_key: inputs}).reshape_as(inputs)
        actual = model._run_generated_input_local_transformer(inputs)

        torch.testing.assert_close(actual, expected)

    assert manager.get_cumulative_stats()["graph_hits"] == 3


@pytest.mark.cuda
def test_input_local_transformer_real_bf16_matches_eager_with_padding() -> None:
    pytest.importorskip("vllm.v1.worker.encoder_cudagraph")
    from transformers import Qwen2Config
    from vllm.v1.worker.encoder_cudagraph import EncoderCudaGraphManager

    from vllm_omni.model_executor.models.mimo_audio.mimo_audio_llm import MiMoAudioQwen2Model

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for encoder CUDA graph capture")

    input_key, _ = _mimo_llm_deps()
    torch.manual_seed(42)
    model = _model()
    model.group_size = 4
    model.input_local_config = SimpleNamespace(hidden_size=32)
    transformer_config = Qwen2Config(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
    )
    model.input_local_transformer = MiMoAudioQwen2Model(transformer_config)
    model = model.eval().to(device="cuda", dtype=torch.bfloat16)

    config = SimpleNamespace(
        compilation_config=SimpleNamespace(
            encoder_cudagraph_token_budgets=[4, 16],
            encoder_cudagraph_max_vision_items_per_batch=1,
            encoder_cudagraph_max_frames_per_batch=0,
        ),
        model_config=SimpleNamespace(
            multimodal_config=SimpleNamespace(
                get_limit_per_prompt=lambda _modality: 0,
                mm_encoder_tp_mode="weights",
            )
        ),
        parallel_config=SimpleNamespace(tensor_parallel_size=1),
        scheduler_config=SimpleNamespace(max_num_seqs=4),
    )
    manager = EncoderCudaGraphManager(
        vllm_config=config,
        device=torch.device("cuda"),
        dtype=torch.bfloat16,
        model=model,
    )
    manager.capture(graph_pool=torch.cuda.graph_pool_handle())
    model.set_input_local_transformer_cudagraph_manager(manager)

    for num_groups in (1, 2, 3, 4):
        inputs = torch.randn(
            (num_groups, 4, 32),
            device="cuda",
            dtype=torch.bfloat16,
        )
        expected = model.encoder_eager_forward({input_key: inputs}).reshape_as(inputs)
        actual = model._run_generated_input_local_transformer(inputs)
        torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)

    assert manager.get_cumulative_stats()["graph_hits"] == 4
