"""사전 설계에 따른 회귀 실험입니다. 모델 선택 함수에는 Batch 2를 전달하지 않습니다."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import GridSearchCV, GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from battery_features import FEATURE_COLUMNS


SEED = 42
TARGET_MAPE = 9.1
# 성능을 보기 전에 정한 순서와 묶음입니다. 평균과 중앙값은 각각 비교합니다.
FEATURE_SETS = {"F1_DeltaQ": ["dq_log10_variance"]}
for charge in ("mean", "median"):
    base = ["dq_log10_variance", "QDischarge_delta_100_2", f"chargetime_{charge}_2_100"]
    policy = base + ["Tavg_mean_2_100", "C1_rate", "switch_soc_pct", "C2_rate"]
    slope = policy + ["QDischarge_slope_2_100"]
    FEATURE_SETS[f"F2_Capacity_Charge_{charge}"] = base
    FEATURE_SETS[f"F3_Temperature_Policy_{charge}"] = policy
    FEATURE_SETS[f"F4_Slope_{charge}"] = slope
    FEATURE_SETS[f"F5_IR_{charge}"] = slope + ["IR_mean_2_100"]


def _check_frame(frame, columns):
    if not columns or len(set(columns)) != len(columns):
        raise ValueError("Feature 목록은 비어 있거나 중복되면 안 됩니다.")
    forbidden = set(columns) - set(FEATURE_COLUMNS)
    if forbidden:
        raise ValueError(f"입력으로 허용하지 않은 컬럼입니다: {sorted(forbidden)}")
    if frame.empty or frame["cell_id"].duplicated().any():
        raise ValueError("비어 있는 데이터 또는 중복 셀 ID가 있습니다.")
    y = frame["cycle_life"].to_numpy(dtype=float)
    x = frame[columns].to_numpy(dtype=float)
    if not np.isfinite(y).all() or (y <= 0).any() or np.isinf(x).any():
        raise ValueError("수명은 유한한 양수여야 하며 Feature에 무한대가 없어야 합니다.")


def prepare_batch1(raw):
    """EDA에서 정한 36개 분석군을 고정합니다. 마지막 용량은 분석군 선정에만 사용합니다."""
    if set(raw["batch"]) != {"Batch 1"}:
        raise ValueError("학습군 선정에는 Batch 1만 전달해야 합니다.")
    audit = raw[["cell_id", "batch", "cycle_life", "last_qd", "early_100_coverage", "quality_flags"]].copy()
    eligible = raw["cycle_life"].gt(0) & np.isfinite(raw["cycle_life"])
    eligible &= raw["last_qd"].gt(0) & raw["last_qd"].le(0.885)
    audit["included"] = eligible
    audit["reason"] = np.where(eligible, "EDA 분석군 유지", "수명 또는 마지막 용량 기준 미충족: 학습 보류")
    main = raw.loc[eligible].copy().reset_index(drop=True)
    if not main["early_100_coverage"].all():
        raise ValueError("학습군에 초기 100사이클이 완전하지 않은 셀이 있습니다. 먼저 기록을 확인하세요.")
    _check_frame(main, FEATURE_COLUMNS)
    if len(main) < 10:
        raise ValueError("정책별 홀드아웃과 CV를 진행하기에 학습 셀이 너무 적습니다.")
    return main, audit


def make_splits(frame, seed=SEED, test_size=0.2):
    """정책 단위로 홀드아웃을 분리한 뒤, 학습 부분에서만 GroupKFold를 구성합니다."""
    _check_frame(frame, FEATURE_COLUMNS)
    if set(frame["batch"]) != {"Batch 1"}:
        raise ValueError("분할 대상은 Batch 1이어야 합니다.")
    groups = frame["policy_group"].astype(str)
    if groups.eq("").any() or groups.nunique() < 6:
        raise ValueError("정책 그룹이 비어 있거나 그룹 수가 충분하지 않습니다.")
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, holdout_idx = next(splitter.split(frame, groups=groups))
    train = frame.iloc[train_idx].reset_index(drop=True)
    heldout = frame.iloc[holdout_idx]
    assert set(train.cell_id).isdisjoint(heldout.cell_id)
    assert set(train.policy_group).isdisjoint(heldout.policy_group)
    n_folds = min(5, train.policy_group.nunique())
    if n_folds < 2:
        raise ValueError("교차검증에 필요한 정책 그룹이 부족합니다.")
    cv_splits = list(GroupKFold(n_splits=n_folds).split(train, groups=train.policy_group))
    fold_rows = []
    seen = np.zeros(len(train), dtype=int)
    for number, (fit_idx, val_idx) in enumerate(cv_splits, 1):
        fit, val = train.iloc[fit_idx], train.iloc[val_idx]
        assert set(fit.cell_id).isdisjoint(val.cell_id)
        assert set(fit.policy_group).isdisjoint(val.policy_group)
        seen[val_idx] += 1
        for role, part in [("fit", fit), ("validation", val)]:
            fold_rows.append({"fold": number, "role": role, "n_cells": len(part),
                              "n_policies": part.policy_group.nunique(),
                              "min_life": part.cycle_life.min(), "max_life": part.cycle_life.max()})
    assert (seen == 1).all()
    split_table = frame[["cell_id", "policy_group", "cycle_life"]].copy()
    split_table["split"] = "B1 CV training pool"
    split_table.loc[split_table.index[holdout_idx], "split"] = "B1 hold-out"
    return {"train_indices": train_idx, "holdout_indices": holdout_idx,
            "cv_splits": cv_splits, "split_table": split_table,
            "fold_table": pd.DataFrame(fold_rows), "seed": seed, "test_size": test_size}


def _model_candidates():
    return {
        "Linear Regression": (LinearRegression(), {}),
        "Ridge": (Ridge(), {"model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]}),
        "Random Forest": (RandomForestRegressor(n_estimators=150, random_state=SEED, n_jobs=1),
                          {"model__max_depth": [2, 4, None], "model__min_samples_leaf": [2, 4]}),
        "SVR": (SVR(kernel="rbf"), {"model__C": [10.0, 100.0, 1000.0],
                                    "model__gamma": ["scale", 0.01, 0.1],
                                    "model__epsilon": [10.0, 30.0]}),
    }


def compare_models(train, cv_splits, n_jobs=1):
    """같은 B1 학습 부분·같은 CV 분할로 4개 모델과 Feature 묶음을 비교합니다."""
    if set(train["batch"]) != {"Batch 1"}:
        raise ValueError("모델 비교에 Batch 2·3을 사용할 수 없습니다.")
    rows, searches = [], {}
    for feature_set, columns in FEATURE_SETS.items():
        _check_frame(train, columns)
        # 학습 fold 전체가 결측이면 대체할 중앙값이 없어 실행을 멈춥니다.
        for fit_idx, _ in cv_splits:
            empty = train.iloc[fit_idx][columns].isna().all()
            if empty.any():
                raise ValueError(f"학습 fold에서 모두 결측인 Feature: {empty[empty].index.tolist()}")
        for model_name, (model, grid) in _model_candidates().items():
            steps = [("imputer", SimpleImputer(strategy="median"))]
            if model_name != "Random Forest":
                steps.append(("scaler", StandardScaler()))
            steps.append(("model", model))
            search = GridSearchCV(Pipeline(steps), grid, cv=cv_splits,
                                  scoring={"mape": "neg_mean_absolute_percentage_error", "mae": "neg_mean_absolute_error"},
                                  refit="mape", n_jobs=n_jobs, error_score="raise", return_train_score=False)
            search.fit(train[columns], train.cycle_life)
            best = search.best_index_
            identifier = f"{model_name}::{feature_set}"
            searches[identifier] = search
            rows.append({"experiment_id": identifier, "model": model_name,
                         "feature_set": feature_set, "n_features": len(columns),
                         "cv_mape_pct": -100 * search.cv_results_["mean_test_mape"][best],
                         "cv_std_pct": 100 * search.cv_results_["std_test_mape"][best],
                         "cv_mae": -search.cv_results_["mean_test_mae"][best],
                         "parameters": json.dumps(search.best_params_, ensure_ascii=False)})
    return pd.DataFrame(rows).sort_values("cv_mape_pct").reset_index(drop=True), searches


def choose_model(comparison, searches, tolerance_pp=1.0):
    """최저 CV MAPE와 1%p 이내면 더 적은 Feature를 선택합니다. B2는 사용하지 않습니다."""
    if tolerance_pp < 0:
        raise ValueError("단순 모델 허용 오차는 0 이상이어야 합니다.")
    minimum = comparison.cv_mape_pct.min()
    candidates = comparison.loc[comparison.cv_mape_pct <= minimum + tolerance_pp].copy()
    candidates["model_order"] = candidates.model.map({"Linear Regression": 0, "Ridge": 1, "SVR": 2, "Random Forest": 3})
    row = candidates.sort_values(["n_features", "cv_mape_pct", "model_order", "experiment_id"]).iloc[0]
    choice = row.drop(labels="model_order").to_dict()
    search = searches[row.experiment_id]
    choice.update({"feature_columns": list(FEATURE_SETS[row.feature_set]),
                   "parameters": search.best_params_, "pipeline": search.best_estimator_,
                   "tolerance_pp": tolerance_pp, "minimum_cv_mape_pct": float(minimum),
                   "selection_reason": f"CV 최저 MAPE + {tolerance_pp:g}%p 이내 후보 중 Feature 수 우선, 다음으로 CV MAPE 비교"})
    return choice


def metrics(predictions):
    y = predictions["actual_cycles"].to_numpy(dtype=float)
    pred = predictions["predicted_cycles"].to_numpy(dtype=float)
    if not len(y) or not np.isfinite(y).all() or (y <= 0).any() or not np.isfinite(pred).all():
        raise ValueError("성능 계산에는 양의 실제 수명과 유한한 예측값이 필요합니다.")
    return {"n": len(y), "mape_pct": float(100 * mean_absolute_percentage_error(y, pred)),
            "mae_cycles": float(mean_absolute_error(y, pred)),
            "mean_error_cycles": float(np.mean(pred - y)),
            "nonpositive_predictions": int((pred <= 0).sum())}


def evaluate_predictions(estimator, frame, columns, training_reference, role):
    """오차는 예측−실제입니다. 음수 예측도 숨기거나 임의로 잘라내지 않습니다."""
    _check_frame(frame, columns)
    _check_frame(training_reference, columns)
    prediction = np.asarray(estimator.predict(frame[columns]), dtype=float)
    if not np.isfinite(prediction).all():
        raise ValueError("모델에서 유한하지 않은 예측이 나왔습니다.")
    result = frame[["cell_id", "batch", "policy_group", "quality_flags"]].copy()
    result["role"] = role
    result["actual_cycles"] = frame.cycle_life.to_numpy()
    result["predicted_cycles"] = prediction
    result["error_cycles"] = prediction - result.actual_cycles
    result["absolute_error"] = result.error_cycles.abs()
    result["ape_pct"] = 100 * result.absolute_error / result.actual_cycles
    result["nonpositive_prediction"] = prediction <= 0
    outside = (frame[columns].lt(training_reference[columns].min()) |
               frame[columns].gt(training_reference[columns].max()))
    result["outside_training_feature_range"] = outside.any(axis=1)
    result["out_of_range_features"] = outside.apply(lambda row: ", ".join(row.index[row]), axis=1)
    result["missing_feature_count"] = frame[columns].isna().sum(axis=1)
    result["below_training_life_range"] = frame.cycle_life < training_reference.cycle_life.min()
    result["above_training_life_range"] = frame.cycle_life > training_reference.cycle_life.max()
    return result.reset_index(drop=True)


def cross_validation_predictions(choice, train, cv_splits):
    """선정 모델의 fold별 점수와 OOF 예측을 남깁니다. 평균은 fold 점수의 산술평균입니다."""
    predictions, scores = [], []
    columns = choice["feature_columns"]
    for fold, (fit_idx, val_idx) in enumerate(cv_splits, 1):
        fit, val = train.iloc[fit_idx], train.iloc[val_idx]
        model = clone(choice["pipeline"]).fit(fit[columns], fit.cycle_life)
        pred = evaluate_predictions(model, val, columns, fit, "Train (Batch 1 CV)")
        pred["fold"] = fold
        predictions.append(pred)
        scores.append({"fold": fold, **metrics(pred)})
    score_frame = pd.DataFrame(scores)
    if not np.isclose(score_frame.mape_pct.mean(), choice["cv_mape_pct"], rtol=1e-8, atol=1e-8):
        raise AssertionError("선정 당시 CV와 다시 계산한 CV 점수가 다릅니다.")
    return pd.concat(predictions, ignore_index=True), score_frame


def _choice_signature(choice):
    # fit 결과가 아니라 모델 종류·전처리·설정·Feature 목록이 같은지 확인합니다.
    spec = {"model": choice["model"], "feature_set": choice["feature_set"],
            "columns": choice["feature_columns"], "parameters": choice["parameters"],
            "steps": [(name, type(step).__name__, step.get_params(deep=False))
                      for name, step in choice["pipeline"].steps]}
    return hashlib.sha256(json.dumps(spec, sort_keys=True, default=str).encode()).hexdigest()


def freeze_choice(choice, splits, output_dir):
    """홀드아웃·B2를 평가하기 전에 확정한 설정을 기록합니다."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {key: value for key, value in choice.items() if key != "pipeline"}
    config.update({"selection_signature": _choice_signature(choice),
                   "random_state": splits["seed"], "holdout_group_fraction": splits["test_size"],
                   "cv_folds": len(splits["cv_splits"]), "target": "cycle_life (total cycles)",
                   "feature_window": "summary cycles 2..100, Qdlin cycles 10 and 100",
                   "cohort": "Batch 1: positive label and 0 < last QD <= 0.885 Ah; fixed during prior EDA",
                   "selection_data": "Batch 1 training pool only",
                   "gap_formula": ["Train - Valid", "Valid - Test", "9.1 - Test"],
                   "holdout_cell_ids": splits["split_table"].query("split == 'B1 hold-out'").cell_id.tolist(),
                   "cv_pool_cell_ids": splits["split_table"].query("split == 'B1 CV training pool'").cell_id.tolist()})
    payload = json.dumps(config, ensure_ascii=False, indent=2, allow_nan=False)
    (output_dir / "selected_config.json").write_text(payload, encoding="utf-8")
    return config


def refit_final_model(choice, batch1, output_dir):
    """선정한 설정 그대로 B1 분석군 전체로 다시 학습합니다."""
    if set(batch1.batch) != {"Batch 1"}:
        raise ValueError("최종 재학습 데이터는 Batch 1이어야 합니다.")
    output_dir = Path(output_dir)
    if not (output_dir / "selected_config.json").exists():
        raise RuntimeError("먼저 freeze_choice로 모델 설정을 확정하세요.")
    frozen = json.loads((output_dir / "selected_config.json").read_text(encoding="utf-8"))
    if frozen["selection_signature"] != _choice_signature(choice):
        raise ValueError("고정한 이후 모델 설정이 변경되었습니다. B2 결과를 보고 재선택하면 안 됩니다.")
    expected_cells = set(frozen["cv_pool_cell_ids"] + frozen["holdout_cell_ids"])
    if set(batch1.cell_id) != expected_cells:
        raise ValueError("최종 재학습 셀이 설정 고정 당시의 B1 분석군과 다릅니다.")
    columns = choice["feature_columns"]
    _check_frame(batch1, columns)
    final = clone(choice["pipeline"]).fit(batch1[columns], batch1.cycle_life)
    joblib.dump({"pipeline": final, "feature_columns": columns,
                 "target": "total cycle_life", "training_cell_ids": batch1.cell_id.tolist()},
                output_dir / "final_model.joblib")
    return final


def performance_table(fold_scores, valid_predictions, test_predictions):
    train = float(fold_scores.mape_pct.mean())
    valid, test = metrics(valid_predictions), metrics(test_predictions)
    rows = [
        ("Train (Batch 1 CV)", train, f"B1 학습 부분 {len(fold_scores)}-fold MAPE 산술평균; 튜닝에 사용한 점수"),
        ("Valid (Batch 1 Hold-out)", valid["mape_pct"], f"정책·셀 분리 n={valid['n']}; MAE={valid['mae_cycles']:.2f}사이클"),
        ("Test (Batch 2)", test["mape_pct"], f"B1 전체 재학습 후 n={test['n']}; MAE={test['mae_cycles']:.2f}사이클"),
        ("Gap (Train-Valid)", train - valid["mape_pct"], "단위 %p; CV - hold-out. 음수이면 뒤쪽 MAPE가 큼"),
        ("Gap (Valid-Test)", valid["mape_pct"] - test["mape_pct"], "단위 %p; hold-out - B2. 학습 표본 수·분포도 다르므로 과적합 단정 불가"),
        ("Gap (Target-Test)", TARGET_MAPE - test["mape_pct"], "단위 %p; 9.1 - B2. 음수이면 논문 참고값보다 오차가 큼"),
    ]
    return pd.DataFrame(rows, columns=["구분", "MAPE (%)", "비고"])


def error_by_life_band(predictions):
    tagged = predictions.copy()
    tagged["life_band"] = np.select([tagged.actual_cycles < 500, tagged.actual_cycles > 1000],
                                    ["Short (<500)", "Long (>1000)"], default="Middle (500-1000)")
    rows = [{"life_band": label, **metrics(part)} for label, part in tagged.groupby("life_band", sort=False)]
    return pd.DataFrame(rows)


def plot_cv_comparison(comparison):
    import matplotlib.pyplot as plt
    best = comparison.sort_values("cv_mape_pct").drop_duplicates("model").sort_values("cv_mape_pct", ascending=False)
    fig, ax = plt.subplots(figsize=(9, 4.8), layout="constrained")
    ax.barh(best.model, best.cv_mape_pct, xerr=best.cv_std_pct, color="#548777", capsize=4)
    ax.set(xlabel="B1 training-pool CV MAPE (%)", title="Four models: best feature set for each model")
    ax.set_xlim(left=0)
    ax.grid(axis="x", alpha=.2)
    ax.set_axisbelow(True)
    fig.text(.01, -.025, "Error bars: standard deviation across folds (not a confidence interval).", fontsize=9)
    return fig


def plot_predictions(valid_predictions, test_predictions):
    import matplotlib.pyplot as plt
    combined = pd.concat([valid_predictions, test_predictions])
    low = min(0, combined.predicted_cycles.min())
    high = max(combined.actual_cycles.max(), combined.predicted_cycles.max()) * 1.06
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), layout="constrained")
    for ax, (title, part, color, marker) in zip(axes, [
        ("B1 hold-out", valid_predictions, "#548777", "o"),
        ("Batch 2 test", test_predictions, "#B87843", "^")]):
        ax.scatter(part.actual_cycles, part.predicted_cycles, color=color, marker=marker, alpha=.8)
        ax.plot([low, high], [low, high], "--", color="#555555", linewidth=1, label="Ideal prediction")
        ax.set(xlabel="Actual total life (cycles)", ylabel="Predicted total life (cycles)",
               title=f"{title}: n={len(part)}, MAPE={metrics(part)['mape_pct']:.2f}%", xlim=(low, high), ylim=(low, high))
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=.2)
        ax.legend(fontsize=9)
    return fig


def plot_error_analysis(test_predictions):
    import matplotlib.pyplot as plt
    top = test_predictions.nlargest(5, "ape_pct").sort_values("ape_pct")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    axes[0].scatter(test_predictions.actual_cycles, test_predictions.error_cycles, marker="^", color="#B87843", alpha=.8)
    axes[0].axhline(0, color="#555555", linewidth=1)
    axes[0].set(xlabel="Actual total life (cycles)", ylabel="Prediction - actual (cycles)", title="Batch 2: direction of prediction error")
    axes[1].barh(top.cell_id, top.ape_pct, color="#548777")
    axes[1].set(xlabel="Absolute percentage error (%)", title="Batch 2: five largest errors", xlim=(0, None))
    for ax in axes:
        ax.grid(alpha=.2)
        ax.set_axisbelow(True)
    return fig
