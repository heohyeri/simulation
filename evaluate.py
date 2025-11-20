import re
from typing import Dict, List, Tuple, Any
import numpy as np

# ============================================================
# 0. EvalConfig & 어댑터: Env -> 평가용 env(dict)
# ============================================================


class EvalConfig:
    """
    evaluate_mapping이 필요로 하는 최소한의 설정만 담는 Config 래퍼.

    - num_users : 사용자 수 K
    - num_rbs   : 글로벌 RB 개수 R (모든 layer RB 합)
    - N_ant_ap  : 안테나 수 (no-MIMO에서는 1로 둠)
    - T         : 슬롯 길이 [초]
    """

    def __init__(self, num_users: int, num_rbs: int, N_ant_ap: int, T: float):
        self.num_users = num_users
        self.num_rbs = num_rbs
        self.N_ant_ap = N_ant_ap
        self.T = T


def build_eval_env_from_env(env_obj: Any) -> Dict[str, Any]:
    """
    Env(env.py의 클래스 인스턴스)를 받아서
    evaluate_mapping에서 사용할 수 있는 '평가용 env 딕셔너리'로 변환.

    env_obj는 최소한 다음 필드를 가진다고 가정:
      - env_obj.cfg
          .num_users
          .num_layers
          .layer_rb_counts
          .layer_to_base_rb
          .rb_bandwidth
          .layer0_rb
          .T
      - env_obj.snr_linear : (K,) 현재 슬롯의 사용자별 SNR (linear)
      - env_obj.queues     : (K,) 현재 슬롯의 사용자별 큐 길이 (bits)

    반환되는 딕셔너리는 다음 key를 가짐:
      - "config"          : EvalConfig 인스턴스
      - "zf_rates_for_set": callable (rb:int, users:List[int]) -> np.ndarray(len(users),)
      - "rb_overlap_mask" : (R,R) bool 배열
      - "G_of_rb"         : (R,) int 배열 (no-MIMO → 전부 1)
      - "q_backlog_bits"  : (K,) 큐 길이 (bits)
      - "R_min_bps"       : (K,) 최소속도 또는 None
    """

    cfg_sim = env_obj.cfg
    K = cfg_sim.num_users
    num_layers = cfg_sim.num_layers
    layer_rb_counts = cfg_sim.layer_rb_counts  # e.g. [37, 16, 8]
    layer_to_base_rb = cfg_sim.layer_to_base_rb
    rb_bw = cfg_sim.rb_bandwidth
    T = cfg_sim.T

    # ---- 글로벌 RB index 매핑 및 RB별 대역폭 B[rb] ----
    global2li: List[Tuple[int, int]] = []  # g -> (l, i)
    B_per_rb: List[float] = []

    for l in range(num_layers):
        cnt = layer_rb_counts[l]
        for i in range(cnt):
            global2li.append((l, i))
            base_rbs = layer_to_base_rb[l][i]  # 예: [0], [0,1], [0,1,2,3]
            B = len(base_rbs) * rb_bw
            B_per_rb.append(B)

    R = len(global2li)
    B_per_rb = np.asarray(B_per_rb, dtype=float)

    # ---- RB 중첩 마스크 (26-tone base RB 기준) ----
    q = cfg_sim.layer0_rb  # base RB 개수 (26-tone RB 개수)
    base_masks = np.zeros((R, q), dtype=bool)

    for g, (l, i) in enumerate(global2li):
        base_indices = layer_to_base_rb[l][i]
        for b_idx in base_indices:
            base_masks[g, b_idx] = True

    overlap = np.zeros((R, R), dtype=bool)
    for a in range(R):
        for b in range(a + 1, R):
            if np.any(base_masks[a] & base_masks[b]):
                overlap[a, b] = True
                overlap[b, a] = True

    # ---- no-MIMO → 각 RB마다 최대 1명만 허용 ----
    G_of_rb = np.ones(R, dtype=int)

    # ---- 백로그 (큐) ----
    if getattr(env_obj, "queues", None) is not None:
        q_bits = np.asarray(env_obj.queues, dtype=float).reshape(-1)
    else:
        q_bits = None

    # ---- SNR (현재 슬롯) ----
    snr_linear = np.asarray(env_obj.snr_linear, dtype=float).reshape(-1)  # (K,)

    # ---- no-MIMO용 "zf_rates_for_set" 정의 ----
    def zf_rates_for_set(rb: int, users: List[int]) -> np.ndarray:
        """
        원래 MIMO ZF에서는 {users} 집합 전체에 대해 ZF rate를 계산하지만,
        여기서는 no-MIMO라서 각 RB에는 최대 1명만 할당된다고 가정.

        rate_k(rb) = B_rb * log2(1 + SNR_k)

        users 리스트 길이가 0이면 빈 배열, >=1이면 각 user에 대해 단순 계산.
        """
        rates = []
        B_rb = float(B_per_rb[rb])
        for u in users:
            gamma_k = snr_linear[u]
            r = B_rb * np.log2(1.0 + gamma_k)
            rates.append(r)
        return np.asarray(rates, dtype=float)

    # ---- 최소 속도 (QoS) → 여기서는 사용하지 않음 ----
    R_min = None

    # ---- EvalConfig 생성 ----
    eval_cfg = EvalConfig(
        num_users=K,
        num_rbs=R,
        N_ant_ap=1,  # no-MIMO
        T=T,
    )

    eval_env: Dict[str, Any] = {
        "config": eval_cfg,
        "zf_rates_for_set": zf_rates_for_set,
        "rb_overlap_mask": overlap,
        "G_of_rb": G_of_rb,
        "q_backlog_bits": q_bits,
        "R_min_bps": R_min,
    }
    return eval_env


# ---------------- Parsing ----------------
# A) "RB 0 -> users [3,7]"
_PAT_LINE_MU = re.compile(
    r"RB\s*(?P<rb>\d+)\s*->\s*users?\s*\[\s*(?P<users>(?:\d+\s*,\s*)*\d+)?\s*\]",
    re.IGNORECASE,
)
# B) "RB 0 -> user 3"
_PAT_LINE_SU = re.compile(
    r"RB\s*(?P<rb>\d+)\s*->\s*user\s*(?P<user>\d+)", re.IGNORECASE
)
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
def _get_env_views(env: Dict[str, Any], cfg: Any):
    R = cfg.num_rbs
    K = cfg.num_users

    # RB 집합에 대한 rate를 계산하는 콜백 함수
    zf_rates_for_set = env.get("zf_rates_for_set", None)
    if zf_rates_for_set is None:
        raise KeyError(
            "env must provide 'zf_rates_for_set' from env_mimo.sample_env()."
        )

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
def check_constraints(
    mapping: Dict[int, List[int]], env: Dict[str, Any]
) -> Tuple[bool, List[str]]:
    """
    c1) 한 유저는 최대 1개 RB만
    c2) 중첩 RB 금지
    c3) 각 RB에서 동시 사용자 수 ≤ G(rb)
    + 인덱스/형식/줄수 검사
    """
    cfg = env["config"]
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
            msgs.append(
                f"Constraint c3 violated at RB {rb}: {len(users)} users > G({rb})={int(G_of_rb[rb])}."
            )

    # c1) 유저 중복 배정 금지
    seen = {}
    for rb, users in mapping.items():
        for u in users:
            if u in seen:
                msgs.append(
                    f"Constraint c1 violated: user {u} on RB {seen[u]} and RB {rb}."
                )
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
    cfg = env["config"]
    K, R = cfg.num_users, cfg.num_rbs
    zf_rates_for_set, overlap, G_of_rb, q_bits, T, R_min = _get_env_views(env, cfg)

    ok, violations = check_constraints(mapping, env)

    # 🔴 인덱스 에러 방어: 잘못된 RB/유저 인덱스가 있으면 ZF 계산 전에 바로 종료
    has_invalid_index = any(
        ("Invalid RB index" in v) or ("Invalid user index" in v) for v in violations
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
    ok = len(violations) == 0

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
    cfg = env["config"]
    R, K, N = cfg.num_rbs, cfg.num_users, cfg.N_ant_ap
    G_of_rb = env.get("G_of_rb", np.full(R, N, dtype=int))
    overlap = env.get("rb_overlap_mask", np.zeros((R, R), dtype=bool))
    q_bits = env.get("q_backlog_bits", None)

    lines = []
    lines.append(
        f"No-MIMO uplink OFDMA (single-antenna users, N_rx={N}), "
        f"RBs={R}, T={cfg.T*1e3:.2f} ms"
    )
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
    score_str = (
        f"{score/1e6:.3f} Mbps" if obj_str == "rate" else f"{score:.3f} (LogSum)"
    )

    parts = [f"Score (Objective={obj_str}): {score_str}"]
    if eval_result["ok"]:
        parts.append("All constraints satisfied.")
    else:
        for v in eval_result["violations"]:
            parts.append(v)
    return "\n".join(parts)
