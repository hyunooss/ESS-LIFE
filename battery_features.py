"""초기 100사이클에서 배터리 수명 예측용 Feature를 만듭니다.

summary 2~100번과 Qdlin 10·100번 사이클을 사용합니다. 수명과 마지막 용량은
점검용 정보이며 모델 입력에서 제외합니다. 결측 대체는 학습 Pipeline에서 수행합니다.

- 용량·IR·충전 시간의 비유한 값과 0 이하 값은 결측 처리합니다.
- 온도는 비유한 값만 제외합니다. 섭씨 0도는 유지합니다.
- b1c0의 12번, b1c18의 40번 용량 이상점은 조건을 재확인한 뒤 제외합니다.
- 큰 충전 시간은 유지하고 평균·중앙값을 각각 계산합니다.
- ΔQ 분산은 ddof=0이며 0 이하이면 로그를 계산하지 않습니다.

원본 MATLAB 파일은 읽기 전용으로 사용합니다.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import pandas as pd


# 2018-04-03은 VarCharge 셀 2개인 별도 파일입니다. 이 프로젝트의 Batch 3은 04-12입니다.
BATCH_FILES = {
    "Batch 1": "2017-05-12_batchdata_updated_struct_errorcorrect.mat",
    "Batch 2": "2018-02-20_batchdata_updated_struct_errorcorrect.mat",
    "Batch 3": "2018-04-12_batchdata_updated_struct_errorcorrect.mat",
}

FEATURE_COLUMNS = [
    "dq_log10_variance",
    "QDischarge_delta_100_2",
    "chargetime_mean_2_100",
    "chargetime_median_2_100",
    "Tavg_mean_2_100",
    "QDischarge_slope_2_100",
    "IR_mean_2_100",
    "C1_rate",
    "switch_soc_pct",
    "C2_rate",
]

_SUMMARY_FIELDS = ("QDischarge", "IR", "Tavg", "chargetime")
_NONPOSITIVE_INVALID = {"QDischarge", "IR", "chargetime"}
_AUDITED_QD_SPIKES = {"b1c0": (12,), "b1c18": (40,)}
_POLICY_PATTERN = re.compile(
    r"(?P<c1>\d+(?:\.\d+)?)C\((?P<soc>\d+(?:\.\d+)?)%\)"
    r"-(?P<c2>\d+(?:\.\d+)?)C(?P<suffix>.*)"
)


def _vector_length(dataset: h5py.Dataset) -> int:
    """MATLAB의 행 벡터와 열 벡터를 모두 허용합니다."""
    if dataset.ndim == 1:
        return dataset.shape[0]
    if dataset.ndim == 2 and 1 in dataset.shape:
        return dataset.size
    raise ValueError(f"1차원 벡터가 아닙니다: {dataset.name}, shape={dataset.shape}")


def _read_vector(dataset: h5py.Dataset, indices=None, *, numeric=True):
    """필요한 위치만 읽습니다. 전체 원시 시계열을 메모리에 올리지 않습니다."""
    _vector_length(dataset)
    if indices is None:
        values = dataset[()]
    elif dataset.ndim == 1:
        values = dataset[indices]
    elif dataset.shape[0] == 1:
        values = dataset[0, indices]
    else:
        values = dataset[indices, 0]
    return np.asarray(values, dtype=float if numeric else None).reshape(-1)


def _referenced_object(file: h5py.File, dataset: h5py.Dataset, index: int):
    """MATLAB 셀 배열의 해당 참조를 한 개만 읽습니다."""
    reference = _read_vector(dataset, index, numeric=False)[0]
    if not reference:
        raise ValueError(f"비어 있는 HDF5 참조: {dataset.name}[{index}]")
    return file[reference]


def _finite_mean(values: np.ndarray) -> float:
    valid = values[np.isfinite(values)]
    return float(valid.mean()) if valid.size else np.nan


def _parse_policy(policy: str):
    """C1/SOC/C2를 추출하고 접미사를 보존한 정책 그룹명을 만듭니다."""
    normalized = re.sub(r"\s+", "", policy).strip()
    match = _POLICY_PATTERN.fullmatch(normalized)
    if match is None:
        return normalized, (np.nan, np.nan, np.nan)
    c1, soc, c2 = (float(match[key]) for key in ("c1", "soc", "c2"))
    if not (c1 > 0 and c2 > 0 and 0 < soc <= 100):
        return normalized, (np.nan, np.nan, np.nan)
    group = f"{c1:g}C({soc:g}%)-{c2:g}C{match['suffix']}"
    return group, (c1, soc, c2)


def _read_policy(file: h5py.File, batch: h5py.Group, index: int) -> str:
    codes = _read_vector(_referenced_object(file, batch["policy_readable"], index))
    if not np.all(np.isfinite(codes)) or not np.all(codes == np.floor(codes)):
        raise ValueError(f"정책 문자열의 문자 코드가 잘못되었습니다: cell index {index}")
    return "".join(chr(int(code)) for code in codes).rstrip("\x00")


def _delta_q_features(file, batch, index, summary_cycles, flags):
    """동일 셀의 고정 전압 격자에서 Qd100 - Qd10을 계산합니다."""
    result = {
        "dq_log10_variance": np.nan,
        "dq_variance_Ah2": np.nan,
        "dq_valid_grid_n": 0,
        "dq_grid_lengths_equal": False,
        "dq_voltage_monotonic": False,
    }
    cycle_group = _referenced_object(file, batch["cycles"], index)
    # cycles에는 별도 cycle 라벨이 없습니다. summary의 실제 번호를 찾은 뒤,
    # 모든 cycles 참조 배열의 길이가 summary와 같은지 확인하여 대응시킵니다.
    for name, dataset in cycle_group.items():
        if _vector_length(dataset) != len(summary_cycles):
            raise ValueError(f"summary와 cycles 길이가 다릅니다: {dataset.name}")
    positions = {cycle: np.flatnonzero(summary_cycles == cycle) for cycle in (10, 100)}
    if any(len(position) != 1 for position in positions.values()):
        flags.append("dq_missing_cycle10_or_100")
        return result

    voltage = _read_vector(_referenced_object(file, batch["Vdlin"], index))
    q10 = _read_vector(_referenced_object(file, cycle_group["Qdlin"], int(positions[10][0])))
    q100 = _read_vector(_referenced_object(file, cycle_group["Qdlin"], int(positions[100][0])))
    equal = len(voltage) == len(q10) == len(q100)
    result["dq_grid_lengths_equal"] = equal
    if not equal:
        flags.append("dq_grid_length_mismatch")
        return result
    valid = np.isfinite(voltage) & np.isfinite(q10) & np.isfinite(q100)
    result["dq_valid_grid_n"] = int(valid.sum())
    if not valid.all():
        flags.append(f"dq_nonfinite_grid_points:{int((~valid).sum())}")
        # 일부 전압 지점만 골라 분산을 계산하면 사전 설계의 전체 격자 정의가 바뀝니다.
        return result
    v = voltage[valid]
    monotonic = len(v) >= 2 and bool(np.all(np.diff(v) > 0) or np.all(np.diff(v) < 0))
    result["dq_voltage_monotonic"] = monotonic
    if not monotonic:
        flags.append("dq_invalid_voltage_grid")
        return result
    # 같은 Vdlin의 같은 위치끼리 빼므로 별도의 보간은 하지 않습니다.
    delta_q = q100[valid] - q10[valid]
    variance = float(np.var(delta_q, ddof=0))
    if not np.isfinite(variance) or variance <= 0:
        flags.append("dq_nonpositive_or_nonfinite_variance")
        return result
    result["dq_variance_Ah2"] = variance
    result["dq_log10_variance"] = float(np.log10(variance))
    return result


def _extract_cell(file, batch, batch_name, index):
    cell_id = f"b{batch_name[-1]}c{index}"
    flags = []
    summary = _referenced_object(file, batch["summary"], index)
    cycles = _read_vector(summary["cycle"])
    if (
        cycles.size == 0
        or not np.all(np.isfinite(cycles))
        or not np.all(cycles == np.floor(cycles))
        or np.any(cycles < 1)
        or np.any(np.diff(cycles) <= 0)
    ):
        raise ValueError(f"{cell_id}: 사이클 번호는 양의 정수이며 중복 없이 증가해야 합니다.")
    for field in _SUMMARY_FIELDS:
        if _vector_length(summary[field]) != len(cycles):
            raise ValueError(f"{cell_id}: cycle과 {field} 배열 길이가 다릅니다.")

    positions = np.flatnonzero((cycles >= 2) & (cycles <= 100))
    early_cycles = cycles[positions]
    complete = np.array_equal(early_cycles, np.arange(2, 101))
    if not complete:
        flags.append("incomplete_cycles2_100")
    target_values = _read_vector(_referenced_object(file, batch["cycle_life"], index))
    if target_values.size != 1:
        raise ValueError(f"{cell_id}: 제공된 cycle_life가 단일 값이 아닙니다.")
    target = float(target_values[0])
    if not np.isfinite(target) or target <= 0:
        target = np.nan
        flags.append("invalid_cycle_life")
    last_qd = float(_read_vector(summary["QDischarge"], len(cycles) - 1)[0])
    if not np.isfinite(last_qd) or last_qd <= 0:
        last_qd = np.nan
        flags.append("invalid_last_qd")
    policy = _read_policy(file, batch, index)
    policy_group, policy_values = _parse_policy(policy)
    if not np.all(np.isfinite(policy_values)):
        flags.append("unparsed_policy")

    result = {
        "cell_id": cell_id,
        "batch": batch_name,
        "source_file": BATCH_FILES[batch_name],
        "policy": policy,
        "policy_group": policy_group,
        "cycle_life": target,
        "last_qd": last_qd,
        "early_rows": int(len(early_cycles)),
        "early_100_coverage": bool(complete),
        "cycle_mapping_verified": True,
        "C1_rate": policy_values[0],
        "switch_soc_pct": policy_values[1],
        "C2_rate": policy_values[2],
    }
    clean = {}
    for field in _SUMMARY_FIELDS:
        values = _read_vector(summary[field], positions)
        invalid = ~np.isfinite(values)
        if field in _NONPOSITIVE_INVALID:
            invalid |= values <= 0
        values[invalid] = np.nan
        if invalid.any():
            flags.append(f"{field}_invalid:{int(invalid.sum())}")
        clean[field] = values
        result[f"{field}_early_invalid_n"] = int(invalid.sum())

    excluded_spikes = 0
    qd = clean["QDischarge"]
    for cycle in _AUDITED_QD_SPIKES.get(cell_id, ()):
        where = np.flatnonzero(early_cycles == cycle)
        if not len(where):
            continue
        position = int(where[0])
        if position == 0 or position == len(qd) - 1:
            raise ValueError(f"{cell_id}: 감사 대상 용량 이상점의 이웃이 없습니다.")
        neighbors = qd[[position - 1, position + 1]]
        if not (
            qd[position] > 1.5
            and np.all(np.isfinite(neighbors))
            and np.all((neighbors > 0.8) & (neighbors < 1.2))
            and early_cycles[position - 1] == cycle - 1
            and early_cycles[position + 1] == cycle + 1
        ):
            raise ValueError(f"{cell_id}, cycle {cycle}: 사전 점검에서 확인한 이상점과 원본 값이 다릅니다.")
        qd[position] = np.nan
        excluded_spikes += 1
        flags.append(f"QD_audited_spike_removed:cycle{cycle}")
    result["QD_isolated_spikes_excluded_n"] = excluded_spikes
    result["QDischarge_early_invalid_n"] += excluded_spikes

    for field, values in clean.items():
        valid_count = int(np.isfinite(values).sum())
        result[f"{field}_valid_early_n"] = valid_count
        result[f"{field}_mean_2_100"] = _finite_mean(values)
        if not valid_count:
            flags.append(f"{field}_all_missing")
    valid_qd = np.isfinite(qd)
    result["QDischarge_slope_2_100"] = (
        float(np.polyfit(early_cycles[valid_qd], qd[valid_qd], 1)[0])
        if valid_qd.sum() >= 3 else np.nan
    )
    endpoint_values = {}
    for cycle in (2, 100):
        where = np.flatnonzero(early_cycles == cycle)
        endpoint_values[cycle] = float(qd[where[0]]) if len(where) else np.nan
    result["QDischarge_delta_100_2"] = endpoint_values[100] - endpoint_values[2]
    result["QDischarge_cycle2"] = endpoint_values[2]
    result["QDischarge_cycle100"] = endpoint_values[100]

    charge = clean["chargetime"]
    finite_charge = charge[np.isfinite(charge)]
    result["chargetime_median_2_100"] = float(np.median(finite_charge)) if len(finite_charge) else np.nan
    result["chargetime_gt100_n"] = int(np.sum(finite_charge > 100))
    if result["chargetime_gt100_n"]:
        flags.append(f"chargetime_gt100_retained:{result['chargetime_gt100_n']}")
    result.update(_delta_q_features(file, batch, index, cycles, flags))

    # Batch 1의 탐색적 near-end 코호트 규칙만 마지막 용량을 사용합니다.
    # Batch 2/3은 양의 유효 수명 레이블을 가진 셀을 외부 평가 후보로 유지합니다.
    valid_target = bool(np.isfinite(target) and target > 0)
    result["main_analysis_eligible"] = valid_target and (
        bool(np.isfinite(last_qd) and last_qd <= 0.885) if batch_name == "Batch 1" else True
    )
    result["cohort_rule"] = "finite_positive_target_and_last_qd_le_0.885" if batch_name == "Batch 1" else "finite_positive_target"
    missing = [column for column in FEATURE_COLUMNS if not np.isfinite(result[column])]
    result["missing_feature_count"] = len(missing)
    result["missing_features"] = "|".join(missing)
    result["quality_flags"] = "|".join(flags) if flags else "ok"
    return result


def validate_feature_table(table: pd.DataFrame) -> None:
    """ID 중복, 수치형 열, 무한대, 입력과 메타데이터의 분리를 점검합니다.

    NaN은 명시적인 측정 결측치이므로 허용합니다. 원본의 구조 오류나 무한대는
    계산을 멈추고 수정해야 하므로 예외를 발생시킵니다.
    """
    required = {"cell_id", "batch", "policy_group", "cycle_life", "last_qd", "main_analysis_eligible", *FEATURE_COLUMNS}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"필수 열이 없습니다: {sorted(missing)}")
    if table.empty:
        raise ValueError("추출한 셀이 없습니다.")
    if table["cell_id"].isna().any() or table["cell_id"].duplicated().any():
        raise ValueError("cell_id가 비어 있거나 중복되었습니다.")
    if {"cycle_life", "last_qd", "main_analysis_eligible"}.intersection(FEATURE_COLUMNS):
        raise ValueError("레이블 또는 코호트 정보가 입력 특성에 포함되었습니다.")
    for column in [*FEATURE_COLUMNS, "cycle_life", "last_qd"]:
        if not pd.api.types.is_numeric_dtype(table[column]):
            raise TypeError(f"수치형이 아닌 열입니다: {column}")
        if np.isinf(table[column].to_numpy(dtype=float)).any():
            raise ValueError(f"무한대가 있는 열입니다: {column}")
    eligible = table["main_analysis_eligible"]
    if not (np.isfinite(table.loc[eligible, "cycle_life"]) & table.loc[eligible, "cycle_life"].gt(0)).all():
        raise ValueError("사용 대상 코호트에 유효하지 않은 레이블이 있습니다.")


def load_feature_table(data_dir, batch_names: Iterable[str] = ("Batch 1",)) -> pd.DataFrame:
    """선택한 배치의 모든 셀을 1행씩 반환합니다. 원본 파일은 수정하지 않습니다.

    예시::

        table = load_feature_table("dataset", ("Batch 1", "Batch 2", "Batch 3"))
        train = table[(table.batch == "Batch 1") & table.main_analysis_eligible].copy()
        X = train[FEATURE_COLUMNS]   # last_qd, cycle_life는 X에 들어가지 않습니다.
        y = train["cycle_life"]

    탈락 셀도 감사할 수 있도록 반환합니다. 모델을 학습/평가하기 전에
    main_analysis_eligible과 early_100_coverage를 확인합니다. 결측치 대체는
    분할 이후 학습 세트에 fit한 Pipeline 안에서 수행합니다.
    """
    if isinstance(batch_names, str):
        raise TypeError("batch_names에는 문자열 하나 대신 ('Batch 1',) 같은 목록을 전달하세요.")
    selected = tuple(batch_names)
    if not selected:
        raise ValueError("최소 한 개 배치를 선택해야 합니다.")
    if len(set(selected)) != len(selected):
        raise ValueError("같은 배치를 두 번 요청하면 cell_id가 중복됩니다.")
    unknown = set(selected).difference(BATCH_FILES)
    if unknown:
        raise ValueError(f"지원하지 않는 배치입니다: {sorted(unknown)}")
    directory = Path(data_dir).expanduser()
    rows = []
    for batch_name in selected:
        path = directory / BATCH_FILES[batch_name]
        if not path.is_file():
            raise FileNotFoundError(f"데이터 파일이 없습니다: {path}")
        with h5py.File(path, "r") as file:
            batch = file["batch"]
            n_cells = _vector_length(batch["summary"])
            for name in ("cycle_life", "policy_readable", "Vdlin", "cycles"):
                if _vector_length(batch[name]) != n_cells:
                    raise ValueError(f"배치의 셀 개수가 일치하지 않습니다: {batch_name}/{name}")
            for index in range(n_cells):
                rows.append(_extract_cell(file, batch, batch_name, index))
    table = pd.DataFrame(rows)
    validate_feature_table(table)
    table.attrs["feature_columns"] = list(FEATURE_COLUMNS)
    table.attrs["feature_cycle_range"] = (2, 100)
    table.attrs["delta_q_cycles"] = (10, 100)
    table.attrs["delta_q_variance_ddof"] = 0
    table.attrs["zero_variance_policy"] = "NaN with quality flag; no log floor"
    table.attrs["cohort_metadata_only"] = ["cycle_life", "last_qd", "main_analysis_eligible"]
    return table
