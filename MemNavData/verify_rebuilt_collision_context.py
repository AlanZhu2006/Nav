"""Recreate the execution mesh context, not a looser collision tolerance.

Habitat 0.3.3 builds its island table before disabling zero-area polygons.
Reloading the resulting mesh can therefore produce different island IDs.
Rebuilding from the hash-bound GLB and saved settings preserves execution
semantics. The rebuilt serialized mesh must match the archived bytes exactly.
This module is only for offline verification; it never runs a policy.
"""
from contextlib import contextmanager
import hashlib
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def require_matching_mesh(archived, rebuilt):
    expected, observed = sha(archived), sha(rebuilt)
    if observed != expected:
        raise ValueError("Rebuilt execution mesh differs from the archived mesh")
    return expected


@contextmanager
def rebuilt_execution_context(asset, asset_sha, archived_mesh, output_mesh):
    import habitat_sim

    if sha(asset) != asset_sha:
        raise ValueError("Frozen scene GLB changed")
    if Path(output_mesh).exists():
        raise ValueError("Do not overwrite a previous verification mesh")
    loaded = habitat_sim.PathFinder()
    if not loaded.load_nav_mesh(str(archived_mesh)):
        raise ValueError("Archived execution mesh cannot be read")
    settings = loaded.nav_mesh_settings
    config = habitat_sim.SimulatorConfiguration()
    config.scene_id = str(asset)
    config.enable_physics = False
    config.create_renderer = False
    agent = habitat_sim.agent.AgentConfiguration()
    agent.sensor_specifications = []
    sim = habitat_sim.Simulator(habitat_sim.Configuration(config, [agent]))
    try:
        if not sim.recompute_navmesh(sim.pathfinder, settings):
            raise ValueError("Cannot reconstruct the original execution mesh")
        if not sim.pathfinder.save_nav_mesh(str(output_mesh)):
            raise ValueError("Cannot serialize the verification mesh")
        mesh_sha = require_matching_mesh(archived_mesh, output_mesh)
        receipt = dict(asset_sha256=asset_sha, navmesh_sha256=mesh_sha,
                       mesh_bytes_identical=True,
                       rebuilt_islands=sim.pathfinder.num_islands,
                       reloaded_islands=loaded.num_islands,
                       collision_tolerance_changed=False,
                       controller_changed=False, policy_executed=False)
        yield sim.pathfinder, receipt
    finally:
        sim.close()
