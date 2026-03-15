import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.io import loadmat


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import leach_python as lp


class LeachPythonTest(unittest.TestCase):
    def test_function_run_is_deterministic(self) -> None:
        def run_once() -> tuple[lp.SimulationReport, np.ndarray, np.ndarray]:
            area, model = lp.standard_model(node_count=20, rounds=30, field_size=20.0)
            rng = np.random.default_rng(11)
            x, y = lp.create_random_locations(model, area, rng)
            sensors = lp.configure_sensors(model, x, y)
            report = lp.run_simulation(area, model, sensors, final_figure=None, show_plot=False, rng=rng)
            return report, x, y

        report_a, x_a, y_a = run_once()
        report_b, x_b, y_b = run_once()

        np.testing.assert_allclose(x_a, x_b)
        np.testing.assert_allclose(y_a, y_b)
        np.testing.assert_array_equal(report_a.dead_nodes, report_b.dead_nodes)
        np.testing.assert_array_equal(report_a.cluster_heads, report_b.cluster_heads)
        np.testing.assert_array_equal(report_a.alive_sensors, report_b.alive_sensors)
        np.testing.assert_allclose(report_a.total_sensor_energy, report_b.total_sensor_energy)

    def test_report_save_writes_expected_fields(self) -> None:
        area, model = lp.standard_model(node_count=12, rounds=8, field_size=12.0)
        rng = np.random.default_rng(5)
        x, y = lp.create_random_locations(model, area, rng)
        sensors = lp.configure_sensors(model, x, y)
        report = lp.run_simulation(area, model, sensors, final_figure=None, show_plot=False, rng=rng)

        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir) / "report.mat"
            lp.save_report(report_path, report, area, model, sensors, x, y, seed=5)
            self.assertTrue(report_path.exists())

            data = loadmat(report_path)
            for key in [
                "Model",
                "Area",
                "X",
                "Y",
                "seed",
                "dead_nodes",
                "cluster_heads",
                "alive_sensors",
                "sensor_energy",
            ]:
                self.assertIn(key, data)

    def test_cli_smoke_run_creates_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir) / "cli_report.mat"
            command = [
                sys.executable,
                str(ROOT / "leach_python.py"),
                "--locations",
                str(ROOT / "Locations.mat"),
                "--rounds",
                "12",
                "--seed",
                "7",
                "--report",
                str(report_path),
            ]
            completed = subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
            self.assertTrue(report_path.exists())
            self.assertIn("Saved report to", completed.stdout)


if __name__ == "__main__":
    unittest.main()
