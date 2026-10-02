"""Small regression checks for evaluation mistakes that can bias results.

Run from the code directory:
    python -m unittest discover -s tests -v
No real battery data or external downloads are needed.
"""

import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import battery_modeling as modeling
from battery_features import FEATURE_COLUMNS


def synthetic_cells(group_sizes=(1, 2, 3) * 4):
    groups = [f"policy_{group}" for group, count in enumerate(group_sizes) for _ in range(count)]
    count = len(groups)
    frame = pd.DataFrame({
        "cell_id": [f"synthetic_{i}" for i in range(count)],
        "batch": "Batch 1",
        "policy_group": groups,
        "cycle_life": np.arange(count, dtype=float) * 20 + 500,
        "last_qd": 0.88,
        "early_100_coverage": True,
        "quality_flags": "ok",
    })
    for column in FEATURE_COLUMNS:
        frame[column] = np.linspace(1.0, 2.0, count)
    return frame


class RecordingImputer(SimpleImputer):
    """Observe the actual inputs received by each cloned CV preprocessing fit."""

    fits = []

    def fit(self, X, y=None):
        super().fit(X, y)
        self.fits.append((tuple(X.index), self.statistics_.copy()))
        return self


class ModelingEvaluationTests(unittest.TestCase):
    def test_metrics_use_each_cells_relative_error(self):
        result = modeling.metrics(pd.DataFrame({
            "actual_cycles": [100.0, 200.0],
            "predicted_cycles": [110.0, 160.0],
        }))
        self.assertEqual(result["n"], 2)
        self.assertAlmostEqual(result["mape_pct"], 15.0)
        self.assertAlmostEqual(result["mae_cycles"], 25.0)
        self.assertAlmostEqual(result["mean_error_cycles"], -15.0)

    def test_gap_sign_and_train_fold_mean_are_explicit(self):
        # Unequal fold sizes distinguish the requested mean of folds (10%)
        # from a pooled, cell-weighted OOF MAPE (14%).
        folds = pd.DataFrame({"mape_pct": [5.0, 15.0], "n": [1, 9]})
        valid = pd.DataFrame({"actual_cycles": [100.0], "predicted_cycles": [120.0]})
        test = pd.DataFrame({"actual_cycles": [100.0], "predicted_cycles": [130.0]})
        values = modeling.performance_table(folds, valid, test).set_index("구분")["MAPE (%)"]
        expected = {
            "Train (Batch 1 CV)": 10.0,
            "Valid (Batch 1 Hold-out)": 20.0,
            "Test (Batch 2)": 30.0,
            "Gap (Train-Valid)": -10.0,
            "Gap (Valid-Test)": -10.0,
            "Gap (Target-Test)": -20.9,
        }
        for label, value in expected.items():
            with self.subTest(label=label):
                self.assertAlmostEqual(values[label], value)

    def test_invalid_targets_raise_and_negative_predictions_are_not_clipped(self):
        result = modeling.metrics(pd.DataFrame({"actual_cycles": [100.0], "predicted_cycles": [-20.0]}))
        self.assertAlmostEqual(result["mape_pct"], 120.0)
        self.assertEqual(result["nonpositive_predictions"], 1)
        for target in [0.0, -1.0, np.nan, np.inf]:
            with self.subTest(target=target), self.assertRaises(ValueError):
                modeling.metrics(pd.DataFrame({"actual_cycles": [target], "predicted_cycles": [100.0]}))

    def test_holdout_and_cv_are_policy_disjoint_with_exact_oof_coverage(self):
        frame = synthetic_cells()
        frame.index = np.arange(len(frame)) * 3 + 10  # Positional splits must survive nondefault row labels.
        splits = modeling.make_splits(frame)
        train_positions = splits["train_indices"]
        holdout_positions = splits["holdout_indices"]
        self.assertFalse(set(train_positions) & set(holdout_positions))
        self.assertEqual(set(train_positions) | set(holdout_positions), set(range(len(frame))))
        train, holdout = frame.iloc[train_positions], frame.iloc[holdout_positions]
        self.assertTrue(set(train.policy_group).isdisjoint(holdout.policy_group))
        self.assertTrue(set(train.cell_id).isdisjoint(holdout.cell_id))
        seen = np.zeros(len(train), dtype=int)
        for fit_positions, validation_positions in splits["cv_splits"]:
            fit, validation = train.iloc[fit_positions], train.iloc[validation_positions]
            self.assertTrue(set(fit.policy_group).isdisjoint(validation.policy_group))
            self.assertTrue(set(fit.cell_id).isdisjoint(validation.cell_id))
            self.assertEqual(set(fit_positions) | set(validation_positions), set(range(len(train))))
            self.assertTrue(set(validation.cell_id).isdisjoint(holdout.cell_id))
            seen[validation_positions] += 1
        np.testing.assert_array_equal(seen, np.ones(len(train), dtype=int))
        reported_holdout = splits["split_table"].query("split == 'B1 hold-out'").cell_id
        self.assertEqual(set(reported_holdout), set(holdout.cell_id))

    def test_metadata_and_future_features_are_rejected_before_prediction(self):
        frame = synthetic_cells()

        class MustNotPredict:
            def predict(self, X):
                raise AssertionError("Forbidden features reached the estimator")

        for forbidden in ["cycle_life", "last_qd", "cell_id", "batch", "knee", "recorded_length", "QD_cycle101"]:
            with self.subTest(column=forbidden), self.assertRaises(ValueError):
                modeling.evaluate_predictions(
                    MustNotPredict(), frame, ["dq_log10_variance", forbidden], frame, "synthetic"
                )

    def test_grid_search_fits_imputer_only_on_each_training_fold(self):
        frame = synthetic_cells((1,) * 6)
        frame["dq_log10_variance"] = [1.0, np.nan, 3.0, 100.0, np.nan, 300.0]
        folds = [(np.array([0, 1, 2]), np.array([3, 4, 5])),
                 (np.array([3, 4, 5]), np.array([0, 1, 2]))]
        RecordingImputer.fits = []
        with patch.object(modeling, "FEATURE_SETS", {"signal": ["dq_log10_variance"]}), \
             patch.object(modeling, "_model_candidates", return_value={"Linear Regression": (LinearRegression(), {})}), \
             patch.object(modeling, "SimpleImputer", RecordingImputer):
            _, searches = modeling.compare_models(frame, folds, n_jobs=1)
        estimator = next(iter(searches.values())).best_estimator_
        self.assertIsInstance(estimator, Pipeline)
        observed = {indices: float(statistics[0]) for indices, statistics in RecordingImputer.fits}
        # A single global median (51.5) in the CV fits would leak the other fold.
        self.assertEqual(set(observed), {(0, 1, 2), (3, 4, 5), tuple(range(6))})
        self.assertAlmostEqual(observed[(0, 1, 2)], 2.0)
        self.assertAlmostEqual(observed[(3, 4, 5)], 200.0)
        self.assertAlmostEqual(observed[tuple(range(6))], 51.5)  # Final refit on the training pool only.

    def _frozen_refit_inputs(self):
        frame = synthetic_cells((2,) * 6)
        columns = ["dq_log10_variance", "QDischarge_delta_100_2"]
        frame[columns[1]] = frame[columns[0]] ** 2
        splits = modeling.make_splits(frame)
        train = frame.iloc[splits["train_indices"]]
        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]).fit(train[columns], train.cycle_life)
        choice = {"model": "Linear Regression", "feature_set": "synthetic",
                  "feature_columns": columns, "parameters": {}, "pipeline": pipeline}
        return frame, splits, choice

    def test_frozen_choice_rejects_mutated_pipeline_or_feature_order(self):
        frame, splits, choice = self._frozen_refit_inputs()
        for mutation in ("model_parameter", "feature_order", "imputer_strategy"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                modeling.freeze_choice(choice, splits, directory)
                changed = copy.deepcopy(choice)
                if mutation == "model_parameter":
                    changed["pipeline"].set_params(model__fit_intercept=False)
                elif mutation == "feature_order":
                    changed["feature_columns"] = changed["feature_columns"][::-1]
                else:
                    changed["pipeline"].set_params(imputer__strategy="mean")
                with self.assertRaises(ValueError):
                    modeling.refit_final_model(changed, frame, directory)
                self.assertFalse((Path(directory) / "final_model.joblib").exists())

    def test_frozen_choice_rejects_changed_training_cohort(self):
        frame, splits, choice = self._frozen_refit_inputs()
        replaced = frame.copy()
        replaced.loc[replaced.index[-1], "cell_id"] = "new_unapproved_cell"
        for changed in [frame.iloc[:-1].copy(), replaced]:
            with self.subTest(ids=changed.cell_id.tolist()), tempfile.TemporaryDirectory() as directory:
                modeling.freeze_choice(choice, splits, directory)
                with self.assertRaises(ValueError):
                    modeling.refit_final_model(choice, changed, directory)
                self.assertFalse((Path(directory) / "final_model.joblib").exists())

    def test_frozen_choice_can_refit_full_approved_cohort_and_round_trip(self):
        frame, splits, choice = self._frozen_refit_inputs()
        # Row order is irrelevant to cohort identity; feature order remains fixed.
        reordered = frame.iloc[::-1].reset_index(drop=True)
        with tempfile.TemporaryDirectory() as directory:
            modeling.freeze_choice(choice, splits, directory)
            final = modeling.refit_final_model(choice, reordered, directory)
            saved = joblib.load(Path(directory) / "final_model.joblib")
            self.assertEqual(set(saved["training_cell_ids"]), set(frame.cell_id))
            self.assertEqual(saved["feature_columns"], choice["feature_columns"])
            self.assertEqual(saved["target"], "total cycle_life")
            expected = final.predict(frame[choice["feature_columns"]])
            np.testing.assert_allclose(saved["pipeline"].predict(frame[choice["feature_columns"]]), expected)
            self.assertTrue(np.isfinite(expected).all())


if __name__ == "__main__":
    unittest.main()
