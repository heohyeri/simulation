import re
from typing import Dict, List, Tuple, Any
import numpy as np
from openai import OpenAI

from config import Config
from env import Env

# 🔹 평가 함수: MIMO용 evaluate.py를 no-MIMO에 맞게 어댑터로 사용
from evaluate import evaluate_mapping, summarize_feedback

import os
from dotenv import load_dotenv

load_dotenv()


# ============================ 설정 ============================

# 사용할 OpenAI 모델 이름
OPENAI_MODEL = "gpt-4.1-mini"  # 필요하면 "gpt-4.1", "gpt-4.1-mini" 등으로 변경

# LLM 출력 파싱용 정규식: "RB g -> user k" 또는 "RB g -> user -1"
_PAT_LINE = re.compile(r"RB\s*(?P<rb>\d+)\s*->\s*user\s*(?P<user>-?\d+)", re.IGNORECASE)


# ===================== RB 인덱스 헬퍼 =====================


def build_global_rb_index(cfg: Config) -> List[Tuple[int, int]]:
    """
    global RB index -> (layer, rb_idx) 매핑 생성.

    예: layer_rb_counts = [37, 16, 8] 이면
        global_rb 0..36  -> (0, 0..36)
        global_rb 37..52 -> (1, 0..15)
        global_rb 53..60 -> (2, 0..7)
    """
    mapping: List[Tuple[int, int]] = []
    for l, cnt in enumerate(cfg.layer_rb_counts):
        for i in range(cnt):
            mapping.append((l, i))
    return mapping


# ===================== 프롬프트 생성 =====================


def summarize_state_for_prompt(state: Dict[str, np.ndarray], cfg: Config) -> str:
    """
    현재 슬롯 상태(state)를 LLM에게 보여줄 요약 문자열로 변환.
    - 시간 t
    - 유저별 거리 / 큐 / SNR(dB)
    - 레이어별 RB 개수
    """
    t = int(state["t"])
    distances = state["distances"]
    queues = state["queues"]
    snr = state["snr"]

    snr_db = 10 * np.log10(snr + 1e-12)

    lines = []
    lines.append(f"[Time slot t = {t}]")
    lines.append(
        f'- Objective: {cfg.objective}  ("rate"=sum-rate, "pf"=proportional fairness)'
    )
    lines.append(f"- Num users K = {cfg.num_users}")
    lines.append(f"- Num layers L = {cfg.num_layers}")
    lines.append(f"- Layer RB counts = {cfg.layer_rb_counts}")
    lines.append("")
    lines.append("Per-user status (index, distance[m], queue[bits], SNR[dB]):")

    for k in range(cfg.num_users):
        lines.append(
            f"  User {k}: d = {distances[k]:.1f} m, "
            f"Q = {queues[k]:.1f} bits, "
            f"SNR = {snr_db[k]:.1f} dB"
        )

    return "\n".join(lines)


def target_format_example(cfg: Config) -> str:
    """
    LLM에게 요구할 출력 형식을 강하게 제한한 문자열.
    설명/해설 금지, 정확히 R_tot줄만 출력하도록 강조.
    중복 유저/겹치는 RB를 쓰면 해당 줄은 버려져 점수 0 처리된다고 경고.
    """
    total_rbs = sum(cfg.layer_rb_counts)
    return f"""IMPORTANT: Output EXACTLY {total_rbs} lines and NOTHING ELSE.
Do NOT include explanations, notes, or extra text.

Each line must follow this format (global RB index g, user index k):
RB g -> user k

Allowed values:
- g: integer in [0..{total_rbs-1}]
- k: -1 (unassigned) or integer in [0..{cfg.num_users-1}]
- Each user may appear at MOST ONCE across all lines (c1). If you repeat a user, that line is discarded and score becomes 0.
- Do NOT assign two RBs that overlap the same base RB. If you do, the latter line is discarded and score becomes 0.

Example (short form; still you must output all {total_rbs} lines):
RB 0 -> user 3
RB 1 -> user -1
RB 2 -> user 1
...
RB {total_rbs-1} -> user -1
"""


def build_history_text(candidates: List[Dict[str, Any]]) -> str:
    """
    OPRO용: 이번 슬롯에서 이전 시도들의 평가 결과를 텍스트로 요약.

    각 candidate에는 "eval_feedback" 문자열이 들어 있다고 가정.
    """
    if not candidates:
        return ""

    lines = []
    lines.append("[Previous attempts and feedback in this time slot]")
    for i, cand in enumerate(candidates):
        lines.append(f"Attempt {i}:")
        # summarize_feedback 결과 문자열을 그대로 사용
        fb = cand.get("eval_feedback", "").strip()
        if fb:
            lines.append(fb)
        # 중복/겹침이 있었다면 명시적으로 경고 추가
        if "c1 violated" in fb or "c2 violated" in fb:
            lines.append(
                "Warning: Duplicate user or overlapping RB detected. "
                "Do NOT repeat users and avoid overlapping base RBs. "
                "Lines that violate c1/c2 are discarded → score drops."
            )
        # sanitize 단계에서 버린 줄 수를 알려줌
        drop_c1 = cand.get("drop_c1", 0)
        drop_c2 = cand.get("drop_c2", 0)
        if drop_c1 or drop_c2:
            lines.append(
                f"Note: {drop_c1} lines dropped due to duplicate users (c1), "
                f"{drop_c2} lines dropped due to overlapping RBs (c2)."
            )
        lines.append("")  # 빈 줄

    return "\n".join(lines)


def build_prompt_for_llm(
    state: Dict[str, np.ndarray],
    cfg: Config,
    history_text: str = "",
) -> str:
    """
    LLM에게 넘길 최종 프롬프트 = 상태 요약 + 출력 형식 설명 + (선택) 이전 시도 피드백.
    """
    header = summarize_state_for_prompt(state, cfg)
    fmt = target_format_example(cfg)

    prompt = header + "\n\n" + fmt

    if history_text:
        # 이전 시도들의 점수/제약 위반 정보를 같이 보여주고,
        # 그걸 참고해서 더 나은 해를 내라고 요청
        prompt += (
            "\n\n"
            + history_text
            + "\n\nUsing the feedback from the previous attempts above, "
            "propose a NEW and IMPROVED allocation for THIS time slot. "
            "Do NOT repeat the same mistakes (constraints c1/c2 violations)."
        )

    return prompt


# ===================== LLM 응답 파싱 =====================


def parse_allocation_from_llm_output(
    text: str,
    cfg: Config,
) -> List[np.ndarray]:
    """
    LLM 응답 텍스트를 Env.step 에서 요구하는 allocation 형식
    (layer별 np.ndarray) 으로 변환한다.

    1) 먼저 global RB index g (0..sum-1) 를 layer, rb_idx로 매핑
    2) 각 라인: "RB g -> user k"
       - k == -1 이면 미할당
       - 0 <= k < num_users 이면 해당 유저 할당
    3) 결과: allocation[l][i] = user index 또는 -1
    """
    total_rbs = sum(cfg.layer_rb_counts)
    global2li = build_global_rb_index(cfg)

    # 초기값: 전부 미할당(-1)
    allocation: List[np.ndarray] = [
        -1 * np.ones(cfg.layer_rb_counts[l], dtype=int) for l in range(cfg.num_layers)
    ]

    mapping: Dict[int, int] = {}  # g -> user
    for rb_str, user_str in _PAT_LINE.findall(text):
        g = int(rb_str)
        k = int(user_str)
        mapping[g] = k

    # 매핑된 g에 대해 allocation 채우기
    for g, k in mapping.items():
        if not (0 <= g < total_rbs):
            # 범위 밖이면 무시 (또는 에러 처리 가능)
            continue
        l, i = global2li[g]
        allocation[l][i] = k

    return allocation


def sanitize_allocation(allocation: List[np.ndarray], cfg: Config) -> Tuple[List[np.ndarray], int, int]:
    """
    간단 보정: 한 유저는 한 번만, base RB 겹치면 후순위 RB를 -1로 비운다.
    (LLM이 c1/c2를 어겼을 때 자동으로 정리)
    """
    cleaned = [arr.copy() for arr in allocation]
    used_users = set()
    used_base = np.zeros(cfg.layer0_rb, dtype=bool)
    global2li = build_global_rb_index(cfg)
    drop_c1 = 0
    drop_c2 = 0

    for g, (l, i) in enumerate(global2li):
        k = int(cleaned[l][i])
        if k < 0:
            continue
        # c1: 이미 배정된 유저면 비움
        if k in used_users:
            cleaned[l][i] = -1
            drop_c1 += 1
            continue
        # c2: base RB 겹치면 비움
        base_indices = cfg.layer_to_base_rb[l][i]
        if any(used_base[b] for b in base_indices):
            cleaned[l][i] = -1
            drop_c2 += 1
            continue
        used_users.add(k)
        for b in base_indices:
            used_base[b] = True

    return cleaned, drop_c1, drop_c2


def allocation_to_mapping(
    allocation: List[np.ndarray],
    cfg: Config,
) -> Dict[int, List[int]]:
    """
    Env.step()에 주는 allocation (layer별 배열)을
    evaluate_mapping()이 기대하는 {RB: [users...]} 형식으로 변환.

    - global RB index g: 0..R_tot-1
    - no-MIMO 이므로, 각 RB에는 0명([]) 또는 1명([k])만 허용.
    """
    global2li = build_global_rb_index(cfg)
    mapping: Dict[int, List[int]] = {}

    for g, (l, i) in enumerate(global2li):
        k = int(allocation[l][i])
        if k < 0:
            mapping[g] = []
        else:
            mapping[g] = [k]

    return mapping


def build_adapter_env_for_evaluator(env_obj: Env, cfg: Config) -> Dict[str, Any]:
    """
    현재 Env 인스턴스를 기반으로,
    evaluate_mapping()이 기대하는 env 딕셔너리 형태로 변환.

    필요한 필드:
      - "config": num_users, num_rbs, N_ant_ap, T
      - "zf_rates_for_set": (rb, users[]) -> per-user rate 벡터
      - "rb_overlap_mask": (R,R) bool 배열
      - "G_of_rb": RB별 동시 사용자 상한 (no-MIMO → 전부 1)
      - "q_backlog_bits": 길이 K 배열
      - "R_min_bps": (선택, 여기선 None)
    """

    class EvalConfig:
        pass

    K = env_obj.num_users
    total_rbs = sum(cfg.layer_rb_counts)
    N_ant_ap = getattr(cfg, "num_antennas", 1)

    eval_cfg = EvalConfig()
    eval_cfg.num_users = K
    eval_cfg.num_rbs = total_rbs
    eval_cfg.N_ant_ap = N_ant_ap
    eval_cfg.T = cfg.T

    # --- global RB -> (layer, i) & base RB 집합 생성 ---
    global2li: List[Tuple[int, int]] = []
    base_sets: List[set] = []
    for l, cnt in enumerate(cfg.layer_rb_counts):
        for i in range(cnt):
            global2li.append((l, i))
            base_sets.append(set(cfg.layer_to_base_rb[l][i]))

    R = total_rbs
    overlap = np.zeros((R, R), dtype=bool)
    for a in range(R):
        for b in range(a + 1, R):
            if base_sets[a] & base_sets[b]:
                overlap[a, b] = overlap[b, a] = True

    # no-MIMO: RB당 최대 1명
    G_of_rb = np.ones(R, dtype=int)

    # 큐: bits 단위라고 가정 (Env에서 그대로 가져옴)
    q_bits = env_obj.queues.copy()
    snr_linear = env_obj.snr_linear.copy()
    rb_bw = cfg.rb_bandwidth

    def zf_rates_for_set(rb: int, users: List[int]) -> np.ndarray:
        """
        no-MIMO 환경을 evaluate.py의 zf_rates_for_set 인터페이스에 맞게 래핑.
        - len(users)==0이면 빈 벡터 반환
        - len(users)>=1 이면, 각 유저에 대해 Env와 동일 공식의 rate 계산
        - MU-MIMO는 사용하지 않으므로, 여러 user를 주더라도
          단순히 '각각 단일 유저일 때 rate'로 계산 (어차피 c3로 걸러짐).
        """
        if len(users) == 0:
            return np.zeros(0, dtype=float)

        l, i = global2li[rb]
        base_rbs = cfg.layer_to_base_rb[l][i]
        B_li = len(base_rbs) * rb_bw

        rates = []
        for u in users:
            gamma_k = snr_linear[u]
            r_li_k = B_li * np.log2(1.0 + gamma_k)
            rates.append(r_li_k)
        return np.array(rates, dtype=float)

    adapter_env: Dict[str, Any] = {
        "config": eval_cfg,
        "zf_rates_for_set": zf_rates_for_set,
        "rb_overlap_mask": overlap,
        "G_of_rb": G_of_rb,
        "q_backlog_bits": q_bits,
        "R_min_bps": None,
    }
    return adapter_env


# ===================== OpenAI LLM 호출 =====================


def call_openai_llm(prompt: str, model: str = OPENAI_MODEL) -> str:
    """
    OpenAI Responses API를 사용해 LLM을 호출하고,
    plain text 형태의 출력을 반환.

    환경 변수 OPENAI_API_KEY가 설정되어 있어야 한다.
    """
    client = OpenAI()

    response = client.responses.create(
        model=model,
        input=prompt,
        max_output_tokens=1500,  # 61줄 강제 출력 대비 토큰 여유 확보
    )

    texts: List[str] = []
    for item in response.output[0].content:
        if getattr(item, "type", None) == "output_text":
            txt = getattr(getattr(item, "output_text", None), "text", None)
            if txt:
                texts.append(txt)

    if not texts:
        return str(response)

    return "".join(texts)


# ===================== 시뮬레이션 루프 (OPRO 버전) =====================


def run_llm_ra_episode(
    cfg: Config,
    num_slots: int,
    model: str = OPENAI_MODEL,
    inner_iters: int = 3,  # OPRO: 슬롯당 LLM 시도 횟수
):
    """
    LLM 기반 RA 에피소드 실행 (no-MIMO + Env + evaluate.py + OPRO):

    각 슬롯마다:
      1) 현재 state와 Env를 기반으로 adapter_env 생성 (평가용)
      2) inner_iters번 반복:
         - 이전 시도들의 eval_feedback을 history_text로 만들어 프롬프트에 추가
         - LLM 호출 → allocation 파싱
         - allocation을 mapping으로 변환 → evaluate_mapping(...) 호출
         - eval_feedback(summarize_feedback) 저장
      3) inner_iters개 candidate 중 점수가 가장 높은 allocation을 선택
      4) 그 allocation을 Env.step()에 넣어 실제 큐/채널 업데이트 + reward 계산
      5) 로그 기록
    """
    env = Env(cfg)
    state = env.reset()

    history = []
    total_reward = 0.0

    for t in range(num_slots):
        print(f"\n================ Slot {t} ================")

        # 현재 Env 상태를 기준으로 평가 환경(adapter_env) 구성
        adapter_env = build_adapter_env_for_evaluator(env, cfg)

        # OPRO용: 이번 슬롯에서의 candidate들
        candidates: List[Dict[str, Any]] = []

        for inner in range(inner_iters):
            # 이전 시도들의 feedback을 history_text로 만듦
            history_text = build_history_text(candidates)

            print(f"\n  [Inner iter {inner}] Calling LLM...")
            prompt = build_prompt_for_llm(state, cfg, history_text=history_text)
            llm_output = call_openai_llm(prompt, model=model)

            # LLM 출력에서 allocation 파싱
            allocation_raw = parse_allocation_from_llm_output(llm_output, cfg)
            allocation, drop_c1, drop_c2 = sanitize_allocation(allocation_raw, cfg)

            # evaluate.py용 mapping 생성
            mapping = allocation_to_mapping(allocation, cfg)

            # 평가 (제약/점수) - 제약 위반 시 score=0으로 처리
            eval_result = evaluate_mapping(
                mapping,
                adapter_env,
                objective=cfg.objective,
                zero_on_violation=True,
            )
            eval_feedback = summarize_feedback(eval_result)

            print("    [Eval] " + eval_feedback.replace("\n", "\n    [Eval] "))

            candidates.append(
                {
                    "allocation": allocation,
                    "mapping": mapping,
                    "eval_result": eval_result,
                    "eval_feedback": eval_feedback,
                    "llm_output": llm_output,
                    "drop_c1": drop_c1,
                    "drop_c2": drop_c2,
                }
            )

        # === inner_iters개 중 가장 높은 score를 가진 candidate 선택 ===
        best_idx = 0
        best_score = candidates[0]["eval_result"]["score"]
        for i in range(1, len(candidates)):
            sc = candidates[i]["eval_result"]["score"]
            if sc > best_score:
                best_score = sc
                best_idx = i

        best_cand = candidates[best_idx]
        best_allocation = best_cand["allocation"]
        best_eval = best_cand["eval_result"]

        print(
            f"\n  [Selection] Chosen attempt = {best_idx}, "
            f"score = {best_eval['score']:.4f}"
        )

        # === 선택한 allocation을 Env.step()에 넣어 실제 reward 계산 ===
        try:
            next_state, reward = env.step(best_allocation)
            violated_env = False
            violation_msg_env = ""
        except AssertionError as e:
            violated_env = True
            violation_msg_env = str(e)
            print(f"  [Env] Constraint violated in Env.step: {violation_msg_env}")
            next_state, reward = state, 0.0

        total_reward += reward

        print(f"  [Env] Reward (objective={cfg.objective}) = {reward:.4f}")

        # 로그 기록
        history.append(
            {
                "t": t,
                "state": state,
                "chosen_allocation": best_allocation,
                "chosen_eval_result": best_eval,
                "reward": reward,
                "violated_env": violated_env,
                "violation_msg_env": violation_msg_env,
                "candidates": candidates,
            }
        )

        state = next_state

    avg_reward = total_reward / float(num_slots) if num_slots > 0 else 0.0

    return {
        "total_reward": total_reward,
        "avg_reward": avg_reward,
        "history": history,
    }


# ===================== 실행 예시 (Scenario 1 & 4) =====================


def run_scenario(objective: str, num_slots: int):
    """
    objective = "rate" (Scenario 1) 또는 "pf" (Scenario 4)
    에 대해 LLM-RA 에피소드를 한 번 실행하고 결과를 출력.
    """
    cfg = Config()
    cfg.objective = objective

    print("=" * 60)
    print(f"Running LLM-RA episode: objective = {objective}, num_slots = {num_slots}")
    print("=" * 60)

    result = run_llm_ra_episode(cfg, num_slots=num_slots, model=OPENAI_MODEL)

    print("\n=== Episode summary ===")
    print(f"Objective            : {objective}")
    print(f"Total reward (Env)   : {result['total_reward']:.4f}")
    print(f"Average per slot     : {result['avg_reward']:.4f}")
    num_viol_env = sum(1 for h in result["history"] if h["violated_env"])
    print(f"Slots with Env violation: {num_viol_env}/{num_slots}")

    return result


def main():
    # 예시: 각 시나리오 5슬롯씩만 테스트 (나중에 50, 100으로 늘려도 됨)
    num_slots = 5

    # Scenario 1: sum-rate
    run_scenario("rate", num_slots)

    # Scenario 4: proportional fairness
    run_scenario("pf", num_slots)


if __name__ == "__main__":
    main()
