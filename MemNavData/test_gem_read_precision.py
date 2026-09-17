"""Storage contract, native computation preservation, and lifecycle guards."""
import ast
import hashlib
import inspect
import os
from pathlib import Path
import sys
import textwrap
from types import SimpleNamespace

import pytest
import torch

from NavDP.baselines.memnav.policy_agent import MemNavAgent
from NavDP.baselines.memnav.gem.episodic import EpisodicGEM


@pytest.fixture
def adapter():
    dependency = os.environ.get('LINGBOT_REPO')
    if dependency:
        sys.path.insert(0, dependency)
    pytest.importorskip('lingbot_map.layers.attention')
    from NavDP.baselines.memnav.gem import read_precision
    return read_precision


def model(adapter):
    result = torch.nn.Module()
    result.aggregator = torch.nn.Module()
    aggregate = result.aggregator
    aggregate.use_sdpa = True
    aggregate.kv_cache_manager = None
    aggregate.total_frames_processed = 0
    aggregate.kv_cache = {f'{kind}_{i}': None for i in range(24) for kind in ('k', 'v')}
    aggregate.global_blocks = torch.nn.ModuleList()
    aggregate.frame_blocks = torch.nn.ModuleList()
    for blocks, count in [(aggregate.global_blocks, 24), (aggregate.frame_blocks, 2)]:
        for _ in range(count):
            block = torch.nn.Module()
            block.attn = adapter.SDPAAttention(8, num_heads=1)
            blocks.append(block)
    return result.eval()


def parameter_identity(module):
    return {name: (id(value), hashlib.sha256(value.detach().numpy().tobytes()).hexdigest())
            for name, value in module.named_parameters()}


def test_select_storage_preserves_checkpoint_weights_and_frame_attention(adapter):
    network = model(adapter)
    original = parameter_identity(network)
    frame_methods = [block.attn.forward.__func__ for block in network.aggregator.frame_blocks]
    receipt = adapter.enable_reader_precision(network)
    assert parameter_identity(network) == original
    assert [block.attn.forward.__func__ for block in network.aggregator.frame_blocks] == frame_methods
    assert all(type(block.attn) is adapter.ReadPrecisionAttention for block in network.aggregator.global_blocks)
    assert adapter.enable_reader_precision(network) == receipt
    assert parameter_identity(network) == original


@pytest.mark.parametrize('invalid', ['populated', 'training', 'mixed', 'non_sdpa'])
def test_unknown_or_populated_state_is_not_reinterpreted(adapter, invalid):
    network = model(adapter)
    if invalid == 'populated':
        network.aggregator.kv_cache['k_0'] = torch.ones(1)
        network.aggregator.total_frames_processed = 1
    elif invalid == 'training':
        network.train()
    elif invalid == 'mixed':
        network.aggregator.global_blocks[0].attn.__class__ = adapter.ReadPrecisionAttention
    else:
        network.aggregator.use_sdpa = False
    classes = [type(block.attn) for block in network.aggregator.global_blocks]
    with pytest.raises(ValueError):
        adapter.enable_reader_precision(network)
    assert [type(block.attn) for block in network.aggregator.global_blocks] == classes


def test_native_forward_diff_is_only_new_key_storage(adapter):
    native = ast.parse(textwrap.dedent(inspect.getsource(adapter.SDPAAttention.forward)))
    stored = ast.parse(textwrap.dedent(inspect.getsource(adapter.ReadPrecisionAttention.forward)))

    class RemoveStorageConversion(ast.NodeTransformer):
        count = 0

        def visit_Assign(self, node):
            if (isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
                    and node.value.func.id == '_gem_key_at_read_precision'):
                self.count += 1
                assert len(node.targets) == 1 and node.targets[0].id == 'k'
                return None
            return node

    remove = RemoveStorageConversion()
    stored = remove.visit(stored)
    assert remove.count == 1
    assert ast.dump(stored, include_attributes=False) == ast.dump(native, include_attributes=False)


@pytest.mark.parametrize('mode,storage', [
    ('legacy', 'reader_precision'), ('connected_reciprocal', 'reader_precision'),
    ('native_interval7', 'unknown'),
])
def test_invalid_storage_selection_fails_before_checkpoint_loading(mode, storage):
    with pytest.raises(ValueError, match='storage'):
        MemNavAgent('/missing/checkpoint', '/missing/internnav',
                    memory_mechanism=mode, memory_kv_storage=storage)


def test_agent_binds_the_selected_storage_to_memory():
    agent = object.__new__(MemNavAgent)
    agent.memory_mechanism = 'native_interval7'
    agent.memory_kv_storage = 'reader_precision'
    assert agent.memory.kv_storage == 'reader_precision'
    assert agent.memory.bounded is False


def test_failed_precision_reset_cannot_continue_with_native_storage():
    agent = object.__new__(MemNavAgent)
    agent.S, agent.W = 8, 16
    agent.certified_reference_depth_source = 'online_history'
    agent.certified_eager_depth_cache = agent.depth_observation_only = False
    agent.lb = SimpleNamespace(model=SimpleNamespace(kv_cache_sliding_window=16))
    memory = EpisodicGEM(agent, bounded=False, kv_storage='reader_precision')
    with pytest.raises(ValueError, match='W64'):
        memory.reset()
    assert memory.failure is not None and memory.reader_precision_receipt is None
    for call in (lambda: memory.write(b'image'), memory.read_dense,
                 lambda: memory.read_sparse(b'goal', [])):
        with pytest.raises(RuntimeError, match='reset'):
            call()
