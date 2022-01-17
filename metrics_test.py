from absl.testing import absltest
from metrics import WeightedScalar, MetricAccumulator


class MetricsTest(absltest.TestCase):
    def testMetricAccumulator(self):
        acc: MetricAccumulator = MetricAccumulator.default_config().instantiate()
        acc.update(
            dict(
                a=WeightedScalar(1, 1),
                b=dict(b1=WeightedScalar(2, 6), b2=WeightedScalar(3, 12)),
            )
        )
        acc.update(
            dict(
                a=WeightedScalar(3, 1),
                b=dict(b1=WeightedScalar(12, 24), b2=WeightedScalar(15, 3)),
            )
        )
        self.assertEqual({"a": 2.0, "b": {"b1": 10.0, "b2": 5.4}}, acc.summaries())


if __name__ == "__main__":
    absltest.main()
