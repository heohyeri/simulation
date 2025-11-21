from typing import List, Tuple
import numpy as np

from config import Config
from env import Env


# ===================== RB 인덱스 =====================


def build_global_rb_index(cfg: Config) -> List[Tuple[int, int]]:
    """
    global RB index g  ->  (layer, rb_idx) 로 매핑하는 리스트 생성

    """
    mapping: List[Tuple[int, int]] = []
    for l, cnt in enumerate(cfg.layer_rb_counts):
        for i in range(cnt):
            mapping.append((l, i))
    return mapping


# ===================== Round-Robin =====================


def true_round_robin_allocation(state, cfg: Config) -> List[np.ndarray]:
    """
    현재 슬롯 상태(state)와 Config를 받아서
    Env.step()에 넣을 allocation 생성

      - 슬롯 인덱스 t를 기준으로 "시작 유저"를 정한다.
      - user_order = [ (t + 0) % K, (t + 1) % K, ..., (t + K-1) % K ]
      - RB를 g = 0..(총 RB-1) 순서대로 보면서
        base RB 중복(c2)이 나지 않는 RB만 고려
        아직 어떤 RB도 배정받지 않은 유저를 user_order 순서대로 골라
        해당 RB에 할당
      - 한 유저는 슬롯당 최대 1개 RB만 할당 (c1)
      - SNR / 거리 / 큐 길이는 전혀 보지 않음
    """
    num_users = cfg.num_users
    total_rbs = sum(cfg.layer_rb_counts)
    global2li = build_global_rb_index(cfg)

    # 기본값
    allocation: List[np.ndarray] = [
        -1 * np.ones(cfg.layer_rb_counts[l], dtype=int) for l in range(cfg.num_layers)
    ]

    # 현재 슬롯 인덱스 t
    t = int(state["t"])

    # ---  Round-Robin용 user 순서 ---
    # 슬롯 t에서는 user_order가 한 칸씩 밀린다.
    user_order = [(t + offset) % num_users for offset in range(num_users)]

    # c1: 사용자당 최대 1개 RB
    user_assigned = np.zeros(num_users, dtype=bool)

    # c2: base RB(26-tone) 중복 금지
    used_base_rb = np.zeros(cfg.layer0_rb, dtype=bool)

    # --- RB를 0번부터 순회하면서 할당 ---
    g = 0  # global RB index
    user_ptr = 0  # user_order에서 현재 후보 인덱스

    while g < total_rbs and user_ptr < num_users:
        l, i = global2li[g]
        base_indices = cfg.layer_to_base_rb[l][
            i
        ]  # 이 RB가 차지하는 base RB index 리스트

        # base RB가 이미 사용 중이면 → 이 RB는 건너뛰고 다음 RB로
        if any(used_base_rb[b] for b in base_indices):
            g += 1
            continue

        # 아직 RB를 배정받지 않은 유저를 user_order에서 찾기
        while user_ptr < num_users and user_assigned[user_order[user_ptr]]:
            user_ptr += 1
        if user_ptr >= num_users:
            break  # 더 이상 배정할 유저 없음

        u = int(user_order[user_ptr])  # 선택된 유저 index

        # 실제 할당
        allocation[l][i] = u
        user_assigned[u] = True
        for b in base_indices:
            used_base_rb[b] = True

        g += 1
        user_ptr += 1

    return allocation


# ===================== 에피소드 실행 =====================


def run_round_robin_episode(cfg: Config, num_slots: int):
    """
    True Round-Robin RA 에피소드 실행:
      - env.reset() → 초기 상태
      - 각 슬롯마다 true_round_robin_allocation()으로 RB 할당
      - env.step()으로 next_state, reward 얻기
    """
    env = Env(cfg)
    state = env.reset()

    total_reward = 0.0
    history = []

    for t in range(num_slots):
        print(f"\n================ Slot {t} ================")

        allocation = true_round_robin_allocation(state, cfg)

        try:
            next_state, reward = env.step(allocation)
            violated = False
            violation_msg = ""
        except AssertionError as e:
            violated = True
            violation_msg = str(e)
            print(f"  [Env] Constraint violated: {violation_msg}")
            next_state, reward = state, 0.0

        print(f"  [Env] Reward (objective={cfg.objective}) = {reward:.4f}")

        total_reward += reward
        history.append(
            {
                "t": t,
                "state": state,
                "allocation": allocation,
                "reward": reward,
                "violated": violated,
                "violation_msg": violation_msg,
            }
        )

        state = next_state

    avg_reward = total_reward / float(num_slots) if num_slots > 0 else 0.0

    print("\n=== Episode summary ===")
    print(f"Objective          : {cfg.objective}")
    print(f"Total reward (Env) : {total_reward:.4f}")
    print(f"Average per slot   : {avg_reward:.4f}")
    num_viol = sum(1 for h in history if h["violated"])
    print(f"Slots with violation: {num_viol}/{num_slots}")

    return {
        "total_reward": total_reward,
        "avg_reward": avg_reward,
        "history": history,
    }


# ===================== Scenario 1 & 4 실행 =====================


def run_scenario(objective: str, num_slots: int):

    cfg = Config()
    cfg.objective = objective

    print("=" * 60)
    print(
        f"Running True Round-Robin episode: objective = {objective}, num_slots = {num_slots}"
    )
    print("=" * 60)

    return run_round_robin_episode(cfg, num_slots=num_slots)


def main():
    # 각 시나리오 5 슬롯씩 실행
    num_slots = 5

    run_scenario("rate", num_slots)

    run_scenario("pf", num_slots)


if __name__ == "__main__":
    main()
