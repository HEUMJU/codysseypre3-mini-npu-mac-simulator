"""Mini NPU: pure-Python MAC pattern classifier."""
import json
import math
import os
import re
import time

EPSILON = 1e-9
REPEATS = 10
WARMUP_REPEATS = 3


def normalize_label(value):
    """Convert accepted data labels to the internal Cross/X vocabulary."""
    if not isinstance(value, str):
        return None
    key = value.strip().lower()
    if key in ("+", "cross"):
        return "Cross"
    if key == "x":
        return "X"
    return None


def validate_matrix(matrix, size=None):
    if not isinstance(matrix, list) or not matrix:
        raise ValueError("배열이 비어 있거나 2차원 배열이 아닙니다")
    n = len(matrix)
    if size is not None and n != size:
        raise ValueError("행 크기 불일치: 예상 {}행, 실제 {}행".format(size, n))
    result = []
    for row_index, row in enumerate(matrix, 1):
        if not isinstance(row, list) or len(row) != n:
            raise ValueError("{}행의 열 수가 {}와 다릅니다".format(row_index, n))
        converted = []
        for value in row:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError("숫자가 아니거나 유한하지 않은 값이 있습니다")
            converted.append(float(value))
        result.append(converted)
    return result


def flatten(matrix):
    return [value for row in matrix for value in row]


def mac_2d(pattern, filt):
    n = len(pattern)
    total = 0.0
    for i in range(n):
        for j in range(n):
            total += pattern[i][j] * filt[i][j]
    return total


def mac_flat(pattern, filt):
    total = 0.0
    for i in range(len(pattern)):
        total += pattern[i] * filt[i]
    return total


def classify(score_cross, score_x):
    difference = score_cross - score_x
    if abs(difference) < EPSILON:
        return "UNDECIDED"
    return "Cross" if difference > 0 else "X"


def classify_manual(score_a, score_b):
    """Return the A/B vocabulary required by the interactive mode."""
    difference = score_a - score_b
    if abs(difference) < EPSILON:
        return "판정 불가 (UNDECIDED)"
    return "A" if difference > 0 else "B"


def timed_stats(function, pattern, filt, repeats=REPEATS):
    # Warm the interpreter and caches before collecting the reported samples.
    for _ in range(WARMUP_REPEATS):
        function(pattern, filt)
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        function(pattern, filt)
        samples.append((time.perf_counter() - start) * 1000.0)
    average = sum(samples) / len(samples)
    variance = sum((sample - average) ** 2 for sample in samples) / max(1, len(samples) - 1)
    return average, math.sqrt(variance)


def benchmark(size, repeats=REPEATS):
    cross = generate_pattern(size, "Cross")
    filt = generate_pattern(size, "Cross")
    flat_cross, flat_filt = flatten(cross), flatten(filt)
    t2, sd2 = timed_stats(mac_2d, cross, filt, repeats)
    tf, sdf = timed_stats(mac_flat, flat_cross, flat_filt, repeats)
    return t2, sd2, tf, sdf


def generate_pattern(size, label):
    if size < 1:
        raise ValueError("크기는 1 이상이어야 합니다")
    kind = normalize_label(label)
    if kind is None:
        raise ValueError("패턴 종류는 Cross 또는 X여야 합니다")
    mid = size // 2
    matrix = []
    for i in range(size):
        row = []
        for j in range(size):
            if kind == "Cross":
                value = 1.0 if i == mid or j == mid else 0.0
            else:
                value = 1.0 if i == j or i + j == size - 1 else 0.0
            row.append(value)
        matrix.append(row)
    return matrix


def read_matrix(label, size):
    while True:
        print("{}: {}개 행을 입력하세요 (각 행 {}개 숫자, 공백 구분)".format(label, size, size))
        matrix = []
        invalid = False
        for i in range(size):
            parts = input("  {}행> ".format(i + 1)).split()
            if len(parts) != size:
                print("입력 형식 오류: 각 줄에 {}개의 숫자를 공백으로 구분해 입력하세요.".format(size))
                invalid = True
                break
            try:
                row = [float(part) for part in parts]
            except ValueError:
                print("숫자 파싱 오류: 각 행에 유효한 숫자 {}개를 공백으로 구분해 입력하세요.".format(size))
                invalid = True
                break
            if not all(math.isfinite(value) for value in row):
                print("숫자 입력 오류: NaN이나 무한대가 아닌 유한한 숫자를 입력하세요.")
                invalid = True
                break
            matrix.append(row)
        if not invalid and len(matrix) == size:
            return matrix
        else:
            print("행렬 전체를 처음부터 다시 입력합니다.")


def mode_manual():
    filt_a = read_matrix("필터 A", 3)
    filt_b = read_matrix("필터 B", 3)
    print("필터 A/B 저장 완료")
    pattern = read_matrix("입력 패턴", 3)
    start = time.perf_counter()
    score_a, score_b = mac_2d(pattern, filt_a), mac_2d(pattern, filt_b)
    elapsed = (time.perf_counter() - start) * 1000.0
    decision = classify_manual(score_a, score_b)
    print("점수 A: {:.12g} | 점수 B: {:.12g}".format(score_a, score_b))
    print("연산 시간: {:.6f} ms | 판정: {}".format(elapsed, decision))
    t2, sd2, tf, sdf = benchmark(3)
    print("성능 분석 (워밍업 {}회 후 {}회 평균±표준편차): 3x3 2차원 {:.6f}±{:.6f} ms, 1차원 {:.6f}±{:.6f} ms, MAC 9회".format(
        WARMUP_REPEATS, REPEATS, t2, sd2, tf, sdf))


def load_json(path):
    with open(path, "r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def mode_json(path):
    try:
        data = load_json(path)
    except (OSError, json.JSONDecodeError) as exc:
        print("JSON 로드 실패: {}".format(exc))
        return
    filters = data.get("filters", {}) if isinstance(data, dict) else {}
    patterns = data.get("patterns", {}) if isinstance(data, dict) else {}
    if not isinstance(filters, dict) or not isinstance(patterns, dict):
        print("스키마 오류: filters와 patterns는 객체여야 합니다.")
        return
    outcomes = []
    print("\n[케이스별 판정]")
    for case_id, case in patterns.items():
        reason = None
        match = re.fullmatch(r"size_(\d+)_\d+", str(case_id))
        if not match:
            reason = "키 규칙 오류(size_{N}_{idx} 필요)"
            size = None
        else:
            size = int(match.group(1))
        try:
            if reason:
                raise ValueError(reason)
            if not isinstance(case, dict) or "input" not in case or "expected" not in case:
                raise ValueError("패턴 항목에 input 또는 expected가 없습니다")
            pattern = validate_matrix(case["input"], size)
            filter_group = filters.get("size_{}".format(size))
            if not isinstance(filter_group, dict):
                raise ValueError("size_{} 필터가 없습니다".format(size))
            by_label = {}
            for raw_label, matrix in filter_group.items():
                normalized = normalize_label(raw_label)
                if normalized:
                    by_label[normalized] = validate_matrix(matrix, size)
            if "Cross" not in by_label or "X" not in by_label:
                raise ValueError("필터 라벨 cross/x가 모두 필요합니다")
            score_cross = mac_2d(pattern, by_label["Cross"])
            score_x = mac_2d(pattern, by_label["X"])
            decision = classify(score_cross, score_x)
            expected = normalize_label(case["expected"])
            if expected is None:
                raise ValueError("지원하지 않는 expected 라벨: {!r}".format(case["expected"]))
            status = "PASS" if decision == expected else "FAIL"
            if status == "FAIL":
                reason = "판정 {} != expected {}".format(decision, expected)
            print("{} | Cross={:.12g} X={:.12g} | 판정={} expected={} | {}{}".format(
                case_id, score_cross, score_x, decision, expected, status,
                " ({})".format(reason) if reason else ""))
            outcomes.append((case_id, status, reason))
        except (ValueError, TypeError, KeyError) as exc:
            reason = str(exc)
            print("{} | FAIL ({})".format(case_id, reason))
            outcomes.append((case_id, "FAIL", reason))

    print("\n[성능 분석: 워밍업 {}회 후 {}회 반복 평균±표준편차]".format(WARMUP_REPEATS, REPEATS))
    print("크기(N×N) | 2차원 평균±표준편차(ms) | 1차원 평균±표준편차(ms) | MAC 연산 횟수(N²)")
    for size in (3, 5, 13, 25):
        t2, sd2, tf, sdf = benchmark(size)
        print("{:>2}×{:<2} | {:>10.6f}±{:<10.6f} | {:>10.6f}±{:<10.6f} | {:>16}".format(
            size, size, t2, sd2, tf, sdf, size * size))
    passed = sum(1 for _, status, _ in outcomes if status == "PASS")
    failed = len(outcomes) - passed
    print("\n[결과 요약] 전체 테스트: {} | 통과: {} | 실패: {}".format(len(outcomes), passed, failed))
    if failed:
        print("실패 케이스:")
        for case_id, status, reason in outcomes:
            if status == "FAIL":
                print("- {}: {}".format(case_id, reason or "원인 미상"))
    else:
        print("실패 케이스 없음 (라벨 정규화 및 epsilon 동점 정책 적용)")


def main():
    print("Mini NPU MAC 패턴 판별기")
    print("1) 사용자 입력 3×3\n2) data.json 일괄 분석")
    while True:
        choice = input("선택 (1/2): ").strip()
        if choice in ("1", "2"):
            break
        print("1 또는 2를 입력하세요.")
    if choice == "1":
        mode_manual()
    else:
        default_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data.json")
        path = input("data.json 경로 (Enter는 {}): ".format(default_path)).strip() or default_path
        mode_json(path)


if __name__ == "__main__":
    main()
