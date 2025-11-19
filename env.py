from typing import Dict, List, Tuple
import numpy as np

from config import Config


class Env:
    def __init__(self, config):
        self.cfg = config
        self.num_users = config.num_users
        self.num_layers = config.num_layers
        self.layer_rb_counts = config.layer_rb_counts
        self.layer_to_base_rb = config.layer_to_base_rb

        self.t = 0  # current time slot index
        self.distances = None  # (K,) user distances
        self.queues = None  # (K,) 각 사용자 큐 길이
        self.h = None  # (K, N) channel vectors
        self.snr_linear = None  # (K,) SNR (linear)

    def reset(self):
        """
        simulation initialization and return the initial state.
        """
        # user distances sampling
        self.distances = np.random.uniform(
            self.cfg.dist_min, self.cfg.dist_max, size=self.num_users
        )

        # initial queues sampling
        self.queues = np.random.uniform(
            self.cfg.queue_min, self.cfg.queue_max, size=self.num_users
        )

        self.t = 0

        # 첫 슬롯 채널/SNR 생성
        self._generate_channels_and_snr()

        return self._get_state()

    def step(self, allocation):
        """
        1. allocation 기반 사용자별 rate r_k 계산
        2. 큐 제약을 고려한 effective rate (r_hat_k) 계산
        3. Reward 계산(objective: rate or pf)
        4. 다음 슬롯을 위한 채널 생성
        5. next_state 반환
        """
        # allocation 형태 체크
        assert len(allocation) == self.num_layers, "allocation 길이가 num_layers와 다릅니다."
        for l in range(self.num_layers):
            assert allocation[l].shape[0] == self.layer_rb_counts[l], \
                f"layer {l} 의 RB 개수가 Config와 다릅니다."

        # 제약 조건 체크
        self._check_constraints(allocation)

        
        # 1) 사용자별 rate r_k 계산
        user_rate = np.zeros(self.num_users, dtype=float)  # r_k = Σ r_{l,i}^{(k)}

        for l in range(self.num_layers):
            for i in range(self.layer_rb_counts[l]):
                k = int(allocation[l][i])
                if k < 0:
                    continue  # 비할당 RB

                assert 0 <= k < self.num_users, f"잘못된 user index: {k}"

                # 이 RB의 유효 대역폭 (26/52/106 tone → 1/2/4 개의 layer0 RB 묶음)
                base_rbs = self.layer_to_base_rb[l][i]  # 예: [0,1] or [0,1,2,3]
                B_li = len(base_rbs) * self.cfg.rb_bandwidth

                # SNR (선형) : 하나의 슬롯에서는 사용자별로 고정
                gamma_k = self.snr_linear[k]

                # 데이터 레이트 r_{l,i}^{(k)} [bits/s]
                r_li_k = B_li * np.log2(1.0 + gamma_k)

                user_rate[k] += r_li_k

        # 2) 큐 제약을 고려한 effective rate 및 큐 업데이트
        # r_hat_k = min( r_k, Q_k / T )
        max_rate_from_queue = self.queues / self.cfg.T  # Q_k / T
        effective_rate = np.minimum(user_rate, max_rate_from_queue)

        # 전송된 "양" (패킷/비트 등 단위) = r_hat_k * T
        transmitted_amount = effective_rate * self.cfg.T

        # 새로 도착하는 패킷/비트
        arrivals = np.random.uniform(
            self.cfg.arrival_min, self.cfg.arrival_max, size=self.num_users
        )

        # 큐 업데이트: Q <- max(Q - transmitted, 0) + arrivals
        self.queues = np.maximum(self.queues - transmitted_amount, 0.0) + arrivals

        # 3) 보상 계산 (objective: rate or pf)
        if self.cfg.objective == "rate":
            # Σ_k r_hat_k
            reward = float(np.sum(effective_rate))
        elif self.cfg.objective == "pf":
            # Σ_k log(1 + r_hat_k)
            reward = float(np.sum(np.log1p(effective_rate)))
        else:
            raise ValueError(f"Unknown objective: {self.cfg.objective}")

        # 4) 다음 슬롯을 위한 채널 업데이트
        self.t += 1
        self._generate_channels_and_snr()

        next_state = self._get_state()
        return next_state, reward

    def _check_constraints(self, allocation):
        K = self.num_users

        # ---------- c1: Σ_{(l,i)} x_{l,i}^{(k)} ≤ 1 ----------
        user_counts = np.zeros(K, dtype=int)

        for l in range(self.num_layers):
            for i in range(self.layer_rb_counts[l]):
                k = int(allocation[l][i])
                if k < 0:
                    continue  # 미할당 RB는 무시
                assert 0 <= k < K, f"잘못된 user index: {k}"
                user_counts[k] += 1

        if np.any(user_counts > 1):
            raise AssertionError(
                f"c1 violation: 어떤 사용자는 여러 RB에 동시에 할당되었습니다. counts={user_counts}"
            )

        # ---------- c2: δ( Σ_k x_{l,i}^{(k)} e_{l,i} ) ≼ 1_q ----------
        # base RB(layer0)의 개수만큼 배열 생성
        q = self.cfg.layer0_rb
        used_base_rb = np.zeros(q, dtype=bool)

        for l in range(self.num_layers):
            for i in range(self.layer_rb_counts[l]):
                k = int(allocation[l][i])
                if k < 0:
                    continue  # 미할당 RB는 무시

                base_indices = self.layer_to_base_rb[l][i]  # 이 RB가 차지하는 layer0 index 집합
                for b_idx in base_indices:
                    if used_base_rb[b_idx]:
                        # 이미 사용 중인 base RB를 또 사용 → overlap 발생
                        raise AssertionError(
                            f"c2 violation: base RB index {b_idx} 가 두 개 이상의 RB에 의해 사용됨 "
                            f"(layer={l}, rb={i})"
                        )
                    used_base_rb[b_idx] = True



    def _generate_channels_and_snr(self):
        """
        각 사용자별 채널 벡터 h^(k) 와 SNR Γ^(k) 생성.
        - g^(k) ~ CN(0, I_N)
        - h^(k) = g^(k) / d_k^{γ/2}
        - w^(k) = h^(k) / ||h^(k)|| (MRC)
        - Γ^(k) = P * |w^H h|^2 / σ^2 = P * ||h||^2 / σ^2
        """
        K = self.num_users
        N = self.cfg.num_antennas

        # g^(k) ~ CN(0, I_N)
        real = np.random.standard_normal(size=(K, N))
        imag = np.random.standard_normal(size=(K, N))
        g = (real + 1j * imag) / np.sqrt(2.0)

        # 거리에 따른 pathloss 적용: h = g / d^{γ/2}
        pathloss = np.power(self.distances, self.cfg.pathloss_gamma / 2.0)  # (K,)
        pathloss = pathloss[:, None]  # (K,1)로 브로드캐스팅
        h = g / pathloss

        self.h = h

        # ||h||^2 계산
        h_norm_sq = np.sum(np.abs(h) ** 2, axis=1)  # (K,)

        # SNR (선형)
        self.snr_linear = (self.cfg.tx_power_watt * h_norm_sq) / self.cfg.noise_power_per_rb

    def _get_state(self):
        return {
            "t": np.array(self.t, dtype=int),
            "distances": self.distances.copy(),
            "queues": self.queues.copy(),
            "snr": self.snr_linear.copy(),
        }
