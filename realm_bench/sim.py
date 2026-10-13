"""Minimal Habitat wrapper used for evaluation.

It reproduces the simulator setup used for the paper: habitat-lab 0.1.7
default config with RGB + semantic sensors at 224x224, HFOV 90, camera at
1.25 m. Agents are placed by teleporting (set_agent_state), so no physics or
sliding settings affect the metrics.
"""

import numpy as np
import quaternion  # noqa: F401  (registers np.quaternion)
from habitat import get_config
from habitat.sims import make_sim

from . import constants as C


class BenchSim:
    def __init__(self, scene_glb: str):
        config = get_config()
        config.defrost()
        config.SIMULATOR.SCENE = scene_glb
        config.SIMULATOR.AGENT_0.SENSORS = ["RGB_SENSOR", "SEMANTIC_SENSOR"]
        for sensor in (config.SIMULATOR.RGB_SENSOR, config.SIMULATOR.SEMANTIC_SENSOR):
            sensor.HFOV = C.HFOV
            sensor.WIDTH = C.IMAGE_WIDTH
            sensor.HEIGHT = C.IMAGE_HEIGHT
            sensor.POSITION = list(C.SENSOR_POSITION)
        config.SIMULATOR.FORWARD_STEP_SIZE = C.FORWARD_STEP_SIZE
        config.SIMULATOR.TURN_ANGLE = C.TURN_ANGLE
        config.freeze()
        self.sim = make_sim(id_sim=config.SIMULATOR.TYPE, config=config.SIMULATOR)
        self.sim.reset()
        self._closed = False

    def set_pose(self, position, rotation_wxyz):
        """position: [x, y, z]; rotation_wxyz: quaternion as [w, x, y, z]."""
        q = np.quaternion(*[float(v) for v in rotation_wxyz])
        self.sim.set_agent_state(np.asarray(position, dtype=np.float32), q)

    def render_rgb(self) -> np.ndarray:
        return self.sim.render("rgb")[..., :3]

    def render_semantic(self) -> np.ndarray:
        return self.sim.render("semantic")

    def semantic_object_ids(self) -> set:
        """Instance ids in the scene's semantic annotation (empty if the
        <scene>_semantic.ply / <scene>.house files are missing)."""
        return {int(o.id.split("_")[-1])
                for o in self.sim.semantic_annotations().objects if o is not None}

    def geodesic_distance(self, a, b) -> float:
        return float(self.sim.geodesic_distance(
            np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)))

    def close(self):
        if not self._closed:
            self.sim.close()
            self._closed = True
