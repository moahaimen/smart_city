import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.models.tcn_predictor import PollutionTCN, PredictorBundle


class TCNPredictorTest(unittest.TestCase):
    def test_forward_shape(self) -> None:
        model = PollutionTCN(input_size=8, channel_size=16, num_blocks=3, kernel_size=3, dropout=0.1)
        inputs = torch.randn(5, 12, 8)
        outputs = model(inputs)
        self.assertEqual(tuple(outputs.shape), (5,))

    def test_bundle_save_and_load_round_trip(self) -> None:
        model = PollutionTCN(input_size=8, channel_size=16, num_blocks=2, kernel_size=3, dropout=0.0)
        bundle = PredictorBundle(
            model=model,
            feature_columns=["a", "b", "c", "d", "e", "f", "g", "h"],
            window_size=12,
            feature_mean=np.zeros(8, dtype=np.float32),
            feature_std=np.ones(8, dtype=np.float32),
            device="cpu",
            checkpoint_path=Path("unused.pt"),
            model_config={"channel_size": 16, "num_blocks": 2, "kernel_size": 3, "dropout": 0.0},
        )

        windows = np.random.default_rng(3).normal(size=(4, 12, 8)).astype(np.float32)
        with tempfile.TemporaryDirectory() as tmpdir:
            bundle.checkpoint_path = Path(tmpdir) / "tcn.pt"
            bundle.save()
            restored = PredictorBundle.load(bundle.checkpoint_path)
            np.testing.assert_allclose(bundle.predict(windows), restored.predict(windows), rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
