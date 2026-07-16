from types import MethodType
import unittest

import torch
from torch import nn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from internnav.model.basemodel.memnav.memnav_policy import MemNavNet


def _generator(seed):
    generator = torch.Generator(device='cpu')
    generator.manual_seed(seed)
    return generator


class MemNavDiffusionSamplingTest(unittest.TestCase):
    @staticmethod
    def _core():
        # Bypass the heavy LingBot/novel constructors while exercising the real
        # reverse-diffusion implementation.
        core = MemNavNet.__new__(MemNavNet)
        nn.Module.__init__(core)
        core.predict_size = 3
        core.device = 'cpu'
        core.noise_scheduler = DDPMScheduler(
            num_train_timesteps=4,
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            prediction_type='epsilon',
        )

        def predict_noise(self, noisy, timestep, current_state, revisit, novel, gate):
            del self, timestep, current_state, revisit, gate
            goal_signal = novel[:, :1, :1]
            return torch.zeros_like(noisy) + goal_signal

        core.predict_noise = MethodType(predict_noise, core)
        return core

    @staticmethod
    def _condition(goal_signal):
        batch_size = 2
        return {
            'current_state': torch.zeros(batch_size, 1, 1),
            'revisit': torch.zeros(batch_size, 1, 1),
            'novel': torch.full((batch_size, 1, 1), float(goal_signal)),
            'revisit_gate': torch.zeros(batch_size),
        }

    def test_complete_sampling_is_reproducible_with_paired_randomness(self):
        core = self._core()
        initial = torch.randn(2, 3, 3, generator=_generator(10))
        first = core.sample_actions_from_condition(
            self._condition(0.25),
            initial_noise=initial,
            generator=_generator(11),
        )
        second = core.sample_actions_from_condition(
            self._condition(0.25),
            initial_noise=initial,
            generator=_generator(11),
        )
        torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)

    def test_complete_sampling_changes_when_only_goal_condition_changes(self):
        core = self._core()
        initial = torch.randn(2, 3, 3, generator=_generator(20))
        correct = core.sample_actions_from_condition(
            self._condition(0.0),
            initial_noise=initial,
            generator=_generator(21),
        )
        shuffled = core.sample_actions_from_condition(
            self._condition(0.75),
            initial_noise=initial,
            generator=_generator(21),
        )
        self.assertGreater(float((correct - shuffled).abs().max()), 1e-4)


if __name__ == '__main__':
    unittest.main()
