from typing import Dict, List, Tuple
import numpy as np

from config import Config


class Env:
    def __init__(self, config: Config):
        self.cfg = config
        self.num_users = config.num_users
        self.num_layers = config.num_layers
        self.layer_rb_counts = config.layer_rb_counts
        self.layer_to_base_rb = config.layer_to_base_rb

        # 시뮬레이션 상태 변수
        self.t = 0  # 현재 time slot index
        self.distances = None  # (K,) 사용자 거리
        self.queues = None  # (K,) 각 사용자 큐 길이
        self.h = None  # (K, N) 채널 벡터
        self.snr_linear = None  # (K,) SNR (선형)

    # ------------------------------------------------------------------
    # 공개 API
    # ------------------------------------------------------------------
    def reset(self) -> Dict[str, np.ndarray]:
        """
        시뮬레이션 초기화 및 초기 state 반환.
        """
        # 사용자 거리 샘플링
        self.distances = np.random.uniform(
            self.cfg.dist_min, self.cfg.dist_max, size=self.num_users
        )

        # 초기 큐 샘플링
        self.queues = np.random.uniform(
            self.cfg.queue_min, self.cfg.queue_max, size=self.num_users
        )

        self.t = 0

        # 첫 슬롯 채널/SNR 생성
        self._generate_channels_and_snr()

        return self._get_state()

    def step(self, allocation: List[np.ndarray]) -> Tuple[Dict[str, np.ndarray], float]:
        """
        한 타임 슬롯 진행.

        Parameters
        ----------
        allocation : List[np.ndarray]
            allocation[l][i] = user index (0 ~ K-1), 또는 -1 (해당 RB 비할당)
            - len(allocation) == num_layers
            - allocation[l].shape[0] == layer_rb_counts[l]

        Returns
        -------
        next_state : dict
            다음 슬롯에서의 상태 (거리, 큐, SNR 등)
        reward : float
            선택된 objective (rate / pf)에 따른 보상
        """
        assert len(allocation) == self.num_layers, "allocation 길이가 num_layers와 다릅니다."
        for l in range(self.num_layers):
            assert allocation[l].shape[0] == self.layer_rb_counts[l], \
                f"layer {l} 의 RB 개수가 Config와 다릅니다."

        # --------------------------------------------------------------
        # 1) 현재 슬롯에서 사용자별 rate 계산
        # --------------------------------------------------------------
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

        # --------------------------------------------------------------
        # 2) 큐 제약을 고려한 effective rate 및 큐 업데이트
        # --------------------------------------------------------------
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

        # --------------------------------------------------------------
        # 3) 보상 계산 (objective: rate or pf)
        # --------------------------------------------------------------
        if self.cfg.objective == "rate":
            # Σ_k r_hat_k
            reward = float(np.sum(effective_rate))
        elif self.cfg.objective == "pf":
            # Σ_k log(1 + r_hat_k)
            reward = float(np.sum(np.log1p(effective_rate)))
        else:
            raise ValueError(f"Unknown objective: {self.cfg.objective}")

        # --------------------------------------------------------------
        # 4) 다음 슬롯을 위한 채널 업데이트
        # --------------------------------------------------------------
        self.t += 1
        self._generate_channels_and_snr()

        next_state = self._get_state()
        return next_state, reward

    # ------------------------------------------------------------------
    # 내부 헬퍼 함수
    # ------------------------------------------------------------------
    def _generate_channels_and_snr(self) -> None:
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

    def _get_state(self) -> Dict[str, np.ndarray]:
        """
        학습/실험용 state 반환.

        state 구성:
        - 't'          : 현재 타임 슬롯 (스칼라, np.array 형태)
        - 'distances'  : (K,) 사용자 거리
        - 'queues'     : (K,) 큐 길이
        - 'snr'        : (K,) 현재 슬롯에서의 사용자별 SNR (선형)
        """
        return {
            "t": np.array(self.t, dtype=int),
            "distances": self.distances.copy(),
            "queues": self.queues.copy(),
            "snr": self.snr_linear.copy(),
        }
