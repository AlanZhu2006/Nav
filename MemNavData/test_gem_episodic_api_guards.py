"""Guard tests to run after the frozen experiment jobs release source files."""
import pytest

from NavDP.baselines.memnav.gem.episodic import EpisodicGEM
from NavDP.baselines.memnav.gem.memory import GeometricEpisodicMemory
from NavDP.baselines.memnav.policy_agent import MemNavAgent


@pytest.mark.parametrize('mode', ['native_interval7', 'connected_reciprocal'])
def test_legacy_checkpoint_rejects_incomplete_episodic_state_before_mutation(mode):
    agent=object.__new__(MemNavAgent)
    agent.memory_mechanism=mode
    agent.marker=object()
    before=vars(agent).copy()
    with pytest.raises(NotImplementedError,match='checkpoint'):
        agent.export_episode_state(resident_fields=set())
    with pytest.raises(NotImplementedError,match='checkpoint'):
        agent.restore_episode_state({'marker':'changed'},resident_fields=set())
    assert vars(agent)==before
    legacy=object.__new__(MemNavAgent)
    with pytest.raises(NotImplementedError,match='checkpoint'):
        legacy.restore_episode_state({'memory_mechanism':mode},resident_fields=set())
    assert vars(legacy)=={}


@pytest.mark.parametrize('failed', [
    {'accepted':False,'pnp':{'status':'runtime_exception'}},
    {'accepted':False,'ranked_candidates':[{'error':'depth archive changed'}]},
])
def test_sparse_inference_failure_cannot_become_a_normal_abstention(monkeypatch,failed):
    backend=object.__new__(MemNavAgent)
    memory=EpisodicGEM(backend)
    monkeypatch.setattr(GeometricEpisodicMemory,'read_sparse',lambda *a,**k:failed)
    with pytest.raises(RuntimeError,match='could not be read'):
        memory.read_sparse(b'goal',[])
    assert memory.failure is not None
    with pytest.raises(RuntimeError,match='reset'):
        memory.read_dense()
    with pytest.raises(RuntimeError,match='reset'):
        memory.write(b'image')


def test_geometric_abstention_preserves_healthy_writer(monkeypatch):
    backend=object.__new__(MemNavAgent)
    memory=EpisodicGEM(backend)
    abstention={'accepted':False,'pnp':{'status':'insufficient_inliers'},'ranked_candidates':[]}
    monkeypatch.setattr(GeometricEpisodicMemory,'read_sparse',lambda *a,**k:abstention)
    assert memory.read_sparse(b'goal',[]) is abstention
    assert memory.failure is None


@pytest.mark.parametrize('source', ['canonical', 'route_sparse'])
def test_sparse_cannot_mix_a_different_depth_provider_into_memory(monkeypatch,source):
    backend=object.__new__(MemNavAgent)
    memory=EpisodicGEM(backend)
    called=[]
    monkeypatch.setattr(GeometricEpisodicMemory,'read_sparse',lambda *a,**k:called.append(k))
    with pytest.raises(ValueError,match='archived online-history'):
        memory.read_sparse(b'goal',[],reference_depth_source=source)
    assert called==[] and memory.failure is None


def test_inherited_canonical_depth_entrypoint_cannot_replay_the_stream(monkeypatch):
    backend=object.__new__(MemNavAgent)
    memory=EpisodicGEM(backend)
    backend.memory=memory
    called=[]
    monkeypatch.setattr(GeometricEpisodicMemory,'read_replayed_depth',lambda *a,**k:called.append(a))
    with pytest.raises(RuntimeError,match='canonical replay is incompatible'):
        backend._certified_reference_depth(8)
    assert called==[] and memory.failure is None
