"""특성 추출의 정보 누출과 잘못된 ΔQ 계산을 방지하는 작은 회귀 테스트.

실행: python -m unittest discover -s tests -v
실제 대용량 데이터 대신 같은 참조 구조의 임시 HDF5 파일을 사용한다.
"""

from pathlib import Path
import sys
import tempfile
import unittest

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from battery_features import BATCH_FILES, FEATURE_COLUMNS, load_feature_table


def make_fixture(directory):
    """1000개 전압 지점과 100사이클 이후 기록 하나를 가진 셀을 만든다."""
    path = Path(directory) / BATCH_FILES["Batch 1"]
    with h5py.File(path, "w") as file:
        batch = file.create_group("batch")
        summary = file.create_group("summary0")
        cycles_group = file.create_group("cycles0")
        cycles = np.r_[np.arange(1, 101), 200.0]
        summary.create_dataset("cycle", data=cycles[:, None])
        qd = 1 + cycles * 0.001
        qd[11] = 1.5390544  # b1c0의 사전 점검 이상점 규칙도 충족한다.
        fields = {
            "QDischarge": qd,
            "IR": np.full(101, 0.02),
            "Tavg": np.full(101, 25.0),
            "chargetime": np.full(101, 10.0),
        }
        for name, values in fields.items():
            summary.create_dataset(name, data=values[:, None])
        objects = {
            "summary": summary,
            "cycles": cycles_group,
            "Vdlin": file.create_dataset("voltage", data=np.linspace(3.5, 2.0, 1000)),
            "cycle_life": file.create_dataset("life", data=[1000.0]),
            "policy_readable": file.create_dataset(
                "policy", data=[ord(c) for c in "4.0C(80%)-4C-newstructure"]
            ),
        }
        for name, obj in objects.items():
            reference = batch.create_dataset(name, (1, 1), dtype=h5py.ref_dtype)
            reference[0, 0] = obj.ref
        x = np.linspace(0, 1, 1000)
        other = file.create_dataset("q_other", data=np.zeros(1000))
        q10 = file.create_dataset("q10", data=x)
        q100 = file.create_dataset("q100", data=x - 0.1 * x**2)
        future = file.create_dataset("q_future", data=x)
        references = cycles_group.create_dataset("Qdlin", (1, 101), dtype=h5py.ref_dtype)
        references[0, :] = [other.ref] * 101
        references[0, 9] = q10.ref
        references[0, 99] = q100.ref
        references[0, 100] = future.ref
    return path


class FeatureExtractionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = make_fixture(self.directory.name)

    def load(self):
        return load_feature_table(self.directory.name)

    def test_future_measurements_and_target_do_not_change_input_features(self):
        before = self.load()
        self.assertTrue(before.loc[0, "early_100_coverage"])
        self.assertFalse(before.loc[0, "main_analysis_eligible"])
        self.assertTrue(np.isfinite(before.loc[0, "dq_log10_variance"]))
        with h5py.File(self.path, "r+") as file:
            for field in ("IR", "Tavg", "chargetime"):
                file["summary0"][field][-1, 0] = 98765
            file["summary0"]["QDischarge"][-1, 0] = 0.88
            file["q_future"][...] = 98765
            file["life"][0] = 555
        after = self.load()
        pd.testing.assert_frame_equal(before[FEATURE_COLUMNS], after[FEATURE_COLUMNS])
        self.assertEqual(after.loc[0, "last_qd"], 0.88)
        self.assertEqual(after.loc[0, "cycle_life"], 555)
        self.assertTrue(after.loc[0, "main_analysis_eligible"])

    def test_zero_variance_is_flagged_missing(self):
        with h5py.File(self.path, "r+") as file:
            file["q100"][...] = file["q10"][...]
        row = self.load().iloc[0]
        self.assertTrue(np.isnan(row["dq_log10_variance"]))
        self.assertIn("dq_nonpositive_or_nonfinite_variance", row["quality_flags"])

    def test_nonfinite_voltage_or_capacity_rejects_entire_delta_q(self):
        for dataset, bad_value in (("voltage", np.nan), ("q10", np.inf), ("q100", np.nan)):
            with self.subTest(dataset=dataset):
                with h5py.File(self.path, "r+") as file:
                    original = file[dataset][500]
                    file[dataset][500] = bad_value
                row = self.load().iloc[0]
                self.assertEqual(row["dq_valid_grid_n"], 999)
                self.assertTrue(np.isnan(row["dq_variance_Ah2"]))
                self.assertTrue(np.isnan(row["dq_log10_variance"]))
                self.assertIn("dq_nonfinite_grid_points:1", row["quality_flags"])
                with h5py.File(self.path, "r+") as file:
                    file[dataset][500] = original

    def test_invalid_cycle_labels_raise(self):
        # 중복, 소수, 비유한 값, 음수 번호를 임의의 행 번호로 대체하지 않는다.
        for bad_value in (9.0, 9.5, np.nan, -1.0):
            with self.subTest(cycle=bad_value):
                with h5py.File(self.path, "r+") as file:
                    file["summary0"]["cycle"][9, 0] = bad_value
                with self.assertRaisesRegex(ValueError, "사이클 번호"):
                    self.load()
                with h5py.File(self.path, "r+") as file:
                    file["summary0"]["cycle"][9, 0] = 10.0


if __name__ == "__main__":
    unittest.main()
