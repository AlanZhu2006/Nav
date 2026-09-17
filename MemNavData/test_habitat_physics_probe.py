"""Two focused regression checks for the isolated Bullet diagnostic."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from MemNavData.habitat_physics_probe import make_sim, primitive, proxy, drive_substep, DT, contacts


class PhysicsInitializationTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="cec_bullet_test_")
        cfg = Path(self.temp.name) / "physics.json"
        cfg.write_text(json.dumps({"physics simulator":"bullet", "timestep":DT, "gravity":[0,-9.81,0]}))
        self.sim = make_sim(cfg)

    def tearDown(self):
        self.sim.close()
        self.temp.cleanup()

    def test_static_geometry_is_placed_before_being_frozen(self):
        floor = primitive(self.sim, "cubeSolid", "test_floor", (5,.1,5), (0,-.1,0))
        wall = primitive(self.sim, "cubeSolid", "test_wall", (2,1,.05), (0,1,-1.5))
        np.testing.assert_allclose(list(floor.translation), [0,-.1,0], atol=1e-7)
        np.testing.assert_allclose(list(wall.translation), [0,1,-1.5], atol=1e-7)

    def test_same_forward_command_is_stopped_by_bullet_contact(self):
        primitive(self.sim, "cubeSolid", "test_floor", (5,.1,5), (0,-.1,0))
        wall = primitive(self.sim, "cubeSolid", "test_wall", (2,1,.05), (0,1,-1.5))
        body = proxy(self.sim)
        for _ in range(240):
            drive_substep(self.sim, body, 0, 0)
        encountered = False
        for _ in range(1920):
            drive_substep(self.sim, body, .376, 0)
            encountered |= any(c["obstacle"] for c in contacts(self.sim, body.object_id, {wall.object_id}))
        self.assertTrue(encountered)
        self.assertGreaterEqual(body.translation.z, -1.17)
        self.assertLess(body.translation.z, -1.10)


if __name__ == "__main__":
    unittest.main()
