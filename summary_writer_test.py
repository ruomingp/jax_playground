import tempfile

from absl.testing import absltest
from metrics import WeightedScalar
from summary_writer import SummaryWriter


class SummaryWriterTest(absltest.TestCase):
    def testAddSummary(self):
        dir = tempfile.mkdtemp()
        writer: SummaryWriter = SummaryWriter.default_config().set(name="test", dir=dir).instantiate(parent=None)
        writer(step=100,
               values={
                   "loss": WeightedScalar(mean=3, weight=16),
                   "accuracy": WeightedScalar(mean=0.7, weight=16),
                   "learner": {"learning_rate": 0.1},
               })


if __name__ == "__main__":
    absltest.main()
