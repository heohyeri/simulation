import re
from typing import Dict, List, Tuple, Any
import numpy as np

from env_mimo import MIMOSimConfig, sample_env

# ---------------- Parsing ----------------
# A) "RB 0 -> users [3,7]"
_PAT_LINE_MU = re.compile(
    r"RB\s*(?P<rb>\d+)\s*->\s*users?\s*\[\s*(?P<users>(?:\d+\s*,\s*)*\d+)?\s*\]",
    re.IGNORECASE,
)
# B) "RB 0 -> user 3"
_PAT_LINE_SU = re.compile(r"RB\s*(?P<rb>\d+)\s*->\s*user\s*(?P<user>\d+)", re.IGNORECASE)
# C) "(0,3), (1,7)"
_PAT_TUPL_SU = re.compile(r"\(\s*(?P<rb>\d+)\s*,\s*(?P<user>\d+)\s*\)")

def parse_solution(text: str) -> Dict[int, List[int]]:
    """
    LLM 응답을 {RB: [users...]}로 파싱.
    빈 리스트 [] 허용 (그 RB 미할당).
    """
    mapping: Dict[int, List[int]] = {}

    for rb, users_str in _PAT_LINE_MU.findall(text):
        rb_i = int(rb)
        if not users_str or users_str.strip() == "":
            mapping[rb_i] = []
        else:
            mapping[rb_i] = [int(x) for x in re.split(r"\s*,\s*", users_str)]

    if mapping:
        return mapping

    su_lines = _PAT_LINE_SU.findall(text)
    if su_lines:
        return {int(rb): [int(u)] for rb, u in su_lines}

    su_tuples = _PAT_TUPL_SU.findall(text)
    if su_tuples:
        return {int(rb): [int(u)] for rb, u in su_tuples}

    return {}

# ---------------- Helpers to read env ----------------
def _get_env_views(env: Dict[str, Any], cfg: MIMOSimConfig):
    R = cfg.num_rbs
    K = cfg.num_users

    # ZF 헬퍼 함수 (env_mimo에서 제공)
    zf_rates_for_set = env.get("zf_rates_for_set", None)
    if zf_rates_for_set is None:
        raise KeyError("env must provide 'zf_rates_for_set' from env_mimo.sample_env().")

    # RB 중첩 마스크(없으면 비중첩)
    overlap = env.get("rb_overlap_mask", np.zeros((R, R), dtype=bool))

    # RB별 동시 사용자 상한 G(rb) (없으면 N)
    G_of_rb = env.get("G_of_rb", np.full(R, getattr(cfg, "N_ant_ap", 1), dtype=int))
    G_of_rb = np.asarray(G_of_rb, dtype=int)

    # 백로그, T (없으면 r_hat=r_th)
    q_bits = env.get("q_backlog_bits", None)
    T = getattr(cfg, "T", None)

    # 최소속도(선택)
    R_min = env.get("R_min_bps", None)

    return zf_rates_for_set, overlap, G_of_rb, q_bits, T, R_min

# ---------------- Constraint checks ----------------
def check_constraints(mapping: Dict[int, List[int]], env: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    c1) 한 유저는 최대 1개 RB만
    c2) 중첩 RB 금지
    c3) 각 RB에서 동시 사용자 수 ≤ G(rb)
    + 인덱스/형식/줄수 검사
    """
    cfg: MIMOSimConfig = env["config"]
    K, R = cfg.num_users, cfg.num_rbs
    msgs: List[str] = []

    # 인덱스/타입 검사
    for rb, users in mapping.items():
        if not isinstance(users, list):
            msgs.append(f"RB {rb}: users must be a list.")
            continue
        if not (0 <= rb < R):
            msgs.append(f"Invalid RB index: {rb} (valid 0..{R-1}).")
        for u in users:
            if not (0 <= u < K):
                msgs.append(f"Invalid user index at RB {rb}: {u} (valid 0..{K-1}).")

    # 정확히 R줄 강제 (RB 0..R-1 모두 포함, 빈 리스트 허용)
    if len(mapping) != R or any(rb not in mapping for rb in range(R)):
        msgs.append(f"Provide EXACTLY {R} lines (one for RB 0..{R-1}).")

    zf_rates_for_set, overlap, G_of_rb, *_ = _get_env_views(env, cfg)

    # c3) RB별 동시 사용자 수 제한
    for rb, users in mapping.items():
        if len(users) > int(G_of_rb[rb]):
            msgs.append(f"Constraint c3 violated at RB {rb}: {len(users)} users > G({rb})={int(G_of_rb[rb])}.")

    # c1) 유저 중복 배정 금지
    seen = {}
    for rb, users in mapping.items():
        for u in users:
            if u in seen:
                msgs.append(f"Constraint c1 violated: user {u} on RB {seen[u]} and RB {rb}.")
            else:
                seen[u] = rb

    # c2) 중첩 RB 동시 사용 금지 (그 RB가 비어있지 않을 때만 고려)
    active_rbs = [rb for rb, users in mapping.items() if len(users) > 0]
    for i in range(len(active_rbs)):
        for j in range(i + 1, len(active_rbs)):
            a, b = active_rbs[i], active_rbs[j]
            if overlap.shape == (R, R) and bool(overlap[a, b]):
                msgs.append(f"Constraint c2 violated: RB {a} overlaps with RB {b}.")

    ok = len(msgs) == 0
    return ok, msgs

# ---------------- Scoring ----------------
def evaluate_mapping(
    mapping: Dict[int, List[int]],
    env: Dict[str, Any],
    objective: str,
    *,
    zero_on_violation: bool = False,  # 제약 위반 시 점수 0 처리 옵션
):
    """
    RB×유저 매핑 기반 채점:
      r_th(k) = ∑_{rb} 1[k∈S_rb] * r_k(rb; S_rb)   (ZF로 집합 의존 rate)
      r_hat(k) = min(r_th(k), q_bits(k)/T)        (백로그가 있을 때)
      objective ∈ {"rate","pf"}:
         rate: ∑ r_hat(k),   pf: ∑ log(1 + r_hat(k))
    """
    cfg: MIMOSimConfig = env["config"]
    K, R = cfg.num_users, cfg.num_rbs
    zf_rates_for_set, overlap, G_of_rb, q_bits, T, R_min = _get_env_views(env, cfg)

    ok, violations = check_constraints(mapping, env)

    # 🔴 인덱스 에러 방어: 잘못된 RB/유저 인덱스가 있으면 ZF 계산 전에 바로 종료
    has_invalid_index = any(
        ("Invalid RB index" in v) or ("Invalid user index" in v)
        for v in violations
    )
    if has_invalid_index:
        r_th = np.zeros(K, dtype=float)
        r_hat = r_th.copy()
        # 점수 계산
        if objective == "rate":
            score = float(r_hat.sum())
        elif objective == "pf":
            score = float(np.log1p(r_hat).sum())
        else:
            raise ValueError("objective must be 'rate' or 'pf'")

        ok = False
        if zero_on_violation and not ok:
            score = 0.0

        return {
            "objective": objective,
            "score": score,
            "ok": ok,
            "violations": violations,
            "per_user_r_th_bps": r_th,
            "per_user_r_hat_bps": r_hat,
            "sum_rate_bps": float(r_hat.sum()),
            "mapping": mapping,
            "G_of_rb": np.asarray(G_of_rb).tolist(),
        }


    # --- r_th 계산 ---
    r_th = np.zeros(K, dtype=float)
    for rb in range(R):
        users = mapping.get(rb, [])
        if len(users) == 0:
            continue
        rb_rates = zf_rates_for_set(rb, users)  # shape (len(users),)
        for idx, u in enumerate(users):
            r_th[u] += float(rb_rates[idx])

    # --- r_hat 계산 ---
    if q_bits is not None and T is not None:
        q_bits = np.asarray(q_bits).reshape(-1)
        if q_bits.shape[0] != K:
            violations.append("q_backlog_bits must have shape (K,).")
            r_hat = r_th
        else:
            r_hat = np.minimum(r_th, q_bits / float(T))
    else:
        r_hat = r_th

    # === c4: QoS 최소속도 하드 제약 ===
    if R_min is not None:
        R_min = np.asarray(R_min).reshape(-1)
        if R_min.shape[0] != K:
            violations.append("R_min_bps must have shape (K,).")
        else:
            assigned = {u for rb in range(R) for u in mapping.get(rb, [])}
            for u in assigned:
                if r_hat[u] < R_min[u]:
                    violations.append(
                        f"Constraint c4 violated: user {u} r_hat={r_hat[u]:.2f} < R_min={R_min[u]:.2f} bps."
                    )

    # 점수 계산
    if objective == "rate":
        score = float(r_hat.sum())
    elif objective == "pf":
        score = float(np.log1p(r_hat).sum())
    else:
        raise ValueError("objective must be 'rate' or 'pf'")

    # 최종 ok 재계산
    ok = (len(violations) == 0)

    # 정책: 위반 시 점수 0 처리 옵션
    if zero_on_violation and not ok:
        score = 0.0

    return {
        "objective": objective,
        "score": score,
        "ok": ok,
        "violations": violations,
        "per_user_r_th_bps": r_th,
        "per_user_r_hat_bps": r_hat,
        "sum_rate_bps": float(r_hat.sum()),
        "mapping": mapping,
        "G_of_rb": np.asarray(G_of_rb).tolist(),
    }

# ---------------- Prompt helpers ----------------
def summarize_env_for_prompt(env: Dict[str, Any]) -> str:
    cfg: MIMOSimConfig = env["config"]
    R, K, N = cfg.num_rbs, cfg.num_users, cfg.N_ant_ap
    G_of_rb = env.get("G_of_rb", np.full(R, N, dtype=int))
    overlap = env.get("rb_overlap_mask", np.zeros((R, R), dtype=bool))
    q_bits = env.get("q_backlog_bits", None)

    lines = []
    lines.append(f"MIMO(N={N}), RBs={R}, Users={K}, T={cfg.T*1e3:.2f} ms")
    lines.append("Per-RB user limit G(rb): " + ", ".join(str(int(g)) for g in G_of_rb))
    ov_pairs = [
        f"({i},{j})" for i in range(R) for j in range(i + 1, R) if bool(overlap[i, j])
    ]
    if ov_pairs:
        lines.append("Overlapping RB pairs: " + " ".join(ov_pairs))
    if q_bits is not None:
        lines.append("Backlog-aware scoring enabled (uses q_bits/T).")
    lines.append("Output one line per RB, e.g. 'RB 0 -> users [3,7]' or [] if empty.")
    return "\n".join(lines)

def target_format_example(num_rbs=9, G_of_rb: List[int] = None) -> str:
    if G_of_rb is None:
        G_note = "up to G(rb) users per RB"
    else:
        G_note = "G(rb)=" + ",".join(str(int(g)) for g in G_of_rb)
    return f"""Write EXACTLY {num_rbs} lines in this format (no extra text):

RB 0 -> users [0,1]
RB 1 -> users []
RB 2 -> users [5]
...

Rules:
- RB index: 0..{num_rbs-1}
- User index: 0..K-1
- Exactly {num_rbs} lines total (one per RB).
- Each user can appear at most once across all RBs (c1).
- Do not allocate two overlapping RBs simultaneously (c2).
- Per-RB user limit (c3): {G_note}.
- Use empty list [] if you do not assign any user to an RB.
"""

def summarize_feedback(eval_result: Dict[str, Any]) -> str:
    obj_str = eval_result["objective"]
    score = eval_result["score"]
    score_str = f"{score/1e6:.3f} Mbps" if obj_str == "rate" else f"{score:.3f} (LogSum)"

    parts = [f"Score (Objective={obj_str}): {score_str}"]
    if eval_result["ok"]:
        parts.append("All constraints satisfied.")
    else:
        for v in eval_result["violations"]:
            parts.append(v)
    return "\n".join(parts)

# ---------------- Demo ----------------
def _demo():
    cfg = MIMOSimConfig()
    env = sample_env(cfg, seed=42)

    # 간단 예시 답안: RB마다 최대 G(rb)만큼 연속 사용자 배치 (데모 전용)
    G = env.get("G_of_rb", np.full(cfg.num_rbs, cfg.N_ant_ap, dtype=int))
    mapping = {}
    next_user = 0
    for rb in range(cfg.num_rbs):
        g = int(G[rb])
        users = list(range(next_user, min(next_user + g, cfg.num_users)))
        next_user += g
        mapping[rb] = users

    # 채점
    result_rate = evaluate_mapping(mapping, env, objective="rate", zero_on_violation=False)
    result_pf = evaluate_mapping(mapping, env, objective="pf", zero_on_violation=False)

    print("--- ENV SUMMARY ---")
    print(summarize_env_for_prompt(env))
    print("\n--- Evaluating 'rate' objective ---")
    print(summarize_feedback(result_rate))
    print("\n--- Evaluating 'pf' objective ---")
    print(summarize_feedback(result_pf))

if __name__ == "__main__":
    _demo()
