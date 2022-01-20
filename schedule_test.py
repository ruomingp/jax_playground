import math

import jax
from absl import logging
from absl.testing import absltest
import numpy as np

import schedule
from schedule import as_schedule_fn


class ScheduleTest(absltest.TestCase):
    def testConstant(self):
        value = 3.14
        s = as_schedule_fn(value)
        for step in range(10):
            self.assertEqual(value, s(step))

    def testLinear(self):
        s = schedule.polynomial(begin_step=10, begin_value=1, end_step=20, end_value=2)
        for step in range(30):
            value = s(step)
            if step < 10:
                self.assertEqual(1, value)
            elif step > 20:
                self.assertEqual(2, value)
            else:
                self.assertEqual(step / 10, value)

    def testSqrt(self):
        s = schedule.polynomial(
            power=0.5, begin_step=0, begin_value=0, end_step=100, end_value=10
        )
        for step in range(10):
            value = s(step)
            np.testing.assert_allclose(math.sqrt(step / 100) * 10, value, atol=1e-6)

    def testExponential(self):
        s = schedule.exponential(
            begin_step=0, begin_value=1, end_step=100, end_value=0.01
        )
        self.assertAlmostEqual(1, s(0))
        self.assertAlmostEqual(0.1, s(50))
        self.assertAlmostEqual(0.01, s(100))
        self.assertAlmostEqual(0.01, s(101))

    def testInverseSqrt(self):
        s = schedule.inverse_sqrt
        for step in range(1, 11):
            value = s(step)
            self.assertAlmostEqual(1 / math.sqrt(step), value)

    def testStepwise(self):
        s = jax.jit(schedule.stepwise(start_step=[100, 200], sub=[0.1, 0.01, 0.001]))
        for step in range(0, 300, 50):
            value = s(step)
            if step < 100:
                self.assertEqual(0.1, value)
            elif step < 200:
                self.assertEqual(0.01, value)
            else:
                self.assertEqual(0.001, value)

    def testT5(self):
        s = jax.jit(
            schedule.stepwise(
                start_step=[100],
                sub=[
                    schedule.polynomial(end_step=100, end_value=0.1),
                    lambda step: schedule.inverse_sqrt(step + 100),
                ],
            )
        )
        for step in range(200):
            value = s(step)
            logging.info("step=%s value=%s", step, value)
            if step <= 100:
                self.assertAlmostEqual((step / 100) * 0.1, value)
            else:
                self.assertAlmostEqual(1 / math.sqrt(step), value)


if __name__ == "__main__":
    absltest.main()
