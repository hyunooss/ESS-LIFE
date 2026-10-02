from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from battery_features import load_feature_table, BATCH_FILES, FEATURE_COLUMNS
from battery_modeling import (
    prepare_batch1, make_splits, compare_models, choose_model, freeze_choice,
    cross_validation_predictions, evaluate_predictions, refit_final_model,
    performance_table, metrics, error_by_life_band, plot_cv_comparison,
    plot_predictions, plot_error_analysis,
)


def evaluation_cohort(raw):
    # Feature의 결측은 셀 삭제 이유로 사용하지 않습니다. 학습 Pipeline으로 대체합니다.
    cohort = raw.loc[np.isfinite(raw.cycle_life) & raw.cycle_life.gt(0)].copy().reset_index(drop=True)
    if not cohort.early_100_coverage.all():
        raise ValueError("평가 후보에 초기 100사이클 기록이 불완전한 셀이 있습니다.")
    return cohort


def run_experiment(data_dir, output_dir, evaluate_batch3=False):
    data_dir, output_dir = Path(data_dir).expanduser().resolve(), Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_b1 = load_feature_table(data_dir, ("Batch 1",))
    batch1, cohort_audit = prepare_batch1(raw_b1)
    splits = make_splits(batch1, seed=42, test_size=0.2)
    train = batch1.iloc[splits["train_indices"]].reset_index(drop=True)
    valid = batch1.iloc[splits["holdout_indices"]].reset_index(drop=True)
    print(f"B1 {len(raw_b1)}개 중 {len(batch1)}개 사용: CV 학습 부분 {len(train)}개 / hold-out {len(valid)}개", flush=True)
    comparison, searches = compare_models(train, splits["cv_splits"], n_jobs=1)
    choice = choose_model(comparison, searches, tolerance_pp=1.0)
    freeze_choice(choice, splits, output_dir)
    cv_predictions, fold_scores = cross_validation_predictions(choice, train, splits["cv_splits"])
    valid_predictions = evaluate_predictions(choice["pipeline"], valid, choice["feature_columns"], train,
                                             "Valid (Batch 1 Hold-out)")
    final_model = refit_final_model(choice, batch1, output_dir)

    # 이 지점 이후에만 Batch 2를 읽습니다. 아래 결과를 모델 선택에 되돌려 사용하지 않습니다.
    raw_b2 = load_feature_table(data_dir, ("Batch 2",))
    batch2 = evaluation_cohort(raw_b2)
    test_predictions = evaluate_predictions(final_model, batch2, choice["feature_columns"], batch1,
                                            "Test (Batch 2)")
    performance = performance_table(fold_scores, valid_predictions, test_predictions)
    tables = {
        "feature_table_batch1": raw_b1, "feature_table_batch2": raw_b2,
        "cohort_audit_batch1": cohort_audit, "data_split": splits["split_table"],
        "cv_fold_composition": splits["fold_table"], "model_comparison": comparison,
        "cv_fold_scores": fold_scores, "cv_predictions": cv_predictions,
        "valid_predictions": valid_predictions, "test_predictions_batch2": test_predictions,
        "model_performance": performance, "error_by_life_band": error_by_life_band(test_predictions),
        "largest_errors_batch2": test_predictions.nlargest(5, "ape_pct"),
    }
    if evaluate_batch3:
        raw_b3 = load_feature_table(data_dir, ("Batch 3",))
        batch3 = evaluation_cohort(raw_b3)
        pred3 = evaluate_predictions(final_model, batch3, choice["feature_columns"], batch1, "Test (Batch 3)")
        tables["feature_table_batch3"] = raw_b3
        tables["test_predictions_batch3"] = pred3
        b2_mape, b3_mape = metrics(test_predictions)["mape_pct"], metrics(pred3)["mape_pct"]
        tables["batch3_additional_performance"] = pd.DataFrame([
            {"구분": "Test (Batch 3)", "MAPE (%)": b3_mape, "비고": f"추가 평가 n={len(batch3)}"},
            {"구분": "Gap (Batch2-Batch3)", "MAPE (%)": b2_mape-b3_mape, "비고": "%p; Batch 2 - Batch 3"},
            {"구분": "Gap (Target-Test, Batch 3)", "MAPE (%)": 9.1-b3_mape,
             "비고": "%p; 9.1은 논문 참고값으로 B3 논문 재현 성능은 아님"},
        ])
    for name, table in tables.items():
        table.to_csv(output_dir / f"{name}.csv", index=False, encoding="utf-8-sig")
    for name, fig in [
        ("cv_model_comparison", plot_cv_comparison(comparison)),
        ("predicted_vs_actual", plot_predictions(valid_predictions, test_predictions)),
        ("batch2_errors", plot_error_analysis(test_predictions)),
    ]:
        fig.savefig(output_dir / f"{name}.png", dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
    summary = {"selected_model": choice["model"], "selected_features": choice["feature_set"],
               "feature_columns": choice["feature_columns"], "cv_mape_pct": float(fold_scores.mape_pct.mean()),
               "valid": metrics(valid_predictions), "test_batch2": metrics(test_predictions),
               "batch1_used": len(batch1), "batch1_excluded": len(raw_b1)-len(batch1),
               "prior_test_eda": True, "test_used_for_selection": False,
               "nonpositive_predictions_clipped": False, "data_sources": []}
    for batch in ("Batch 1", "Batch 2") + (("Batch 3",) if evaluate_batch3 else ()):
        source = data_dir / BATCH_FILES[batch]
        summary["data_sources"].append({"batch": batch, "filename": source.name,
                                        "bytes": source.stat().st_size, "mtime_ns": source.stat().st_mtime_ns})
    (output_dir / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(f"선택: {choice['model']} / {choice['feature_set']}")
    print(performance.to_string(index=False))
    print(f"결과 저장: {output_dir}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EDA 설계에 따른 ESS 배터리 수명 회귀 실험")
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent / "dataset")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "results")
    parser.add_argument("--batch3", action="store_true", help="선택한 모델을 변경하지 않고 Batch 3도 추가 평가")
    args = parser.parse_args()
    run_experiment(args.data_dir, args.output_dir, args.batch3)
