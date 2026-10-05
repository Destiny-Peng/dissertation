import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from common import state_delta, metric_to_osc


class TransferContracts(unittest.TestCase):
    def test_se3_target_recovers_cartesian_delta_under_rotation(self):
        pos = np.array([[.4, -.2, .3], [.415, -.202, .29]])
        rot = np.array([[.1, -.2, .4], [.105, -.17, .42]])
        fingers = np.array([[.04, -.04], [.039, -.039]])
        delta = state_delta(pos, rot, fingers)
        restored = Rotation.from_rotvec(delta[:, 3:6]).apply(pos[:-1]) + delta[:, :3]
        np.testing.assert_allclose(restored, pos[1:], atol=1e-12)
        np.testing.assert_allclose(delta[:, 6], [-.001], atol=1e-12)
        cfg = {'output_max': [.05] * 3 + [.5] * 3, 'output_min': [-.05] * 3 + [-.5] * 3}
        raw, clipped = metric_to_osc(delta, pos[:-1], fingers[:-1], np.array([-1.]), cfg)
        np.testing.assert_allclose(raw[:, :3], (pos[1:] - pos[:-1]) / .05, atol=1e-12)
        self.assertEqual(clipped[0, 6], 1.)

    def test_rotation_is_world_left_multiplication(self):
        r0 = Rotation.from_euler('xyz', [.4, -.2, .1])
        dr = Rotation.from_rotvec([.02, -.01, .03])
        r1 = dr * r0
        delta = state_delta(np.zeros((2, 3)), np.stack([r0.as_rotvec(), r1.as_rotvec()]), np.zeros((2, 2)))
        np.testing.assert_allclose(delta[0, 3:6], dr.as_rotvec(), atol=1e-12)

    def test_clipping_preserves_raw_and_gripper_sign(self):
        metric = np.zeros((2, 7))
        metric[:, 0] = [.2, -.2]
        cfg = {'output_max': [.05] * 3 + [.5] * 3, 'output_min': [-.05] * 3 + [-.5] * 3}
        raw, clipped = metric_to_osc(metric, np.zeros((2, 3)), np.full((2, 2), .03), np.array([1., -1.]), cfg)
        np.testing.assert_equal(raw[:, 0], [4., -4.])
        np.testing.assert_equal(clipped[:, 0], [1., -1.])
        np.testing.assert_equal(clipped[:, 6], [-1., 1.])


if __name__ == '__main__':
    unittest.main()
