from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np


@dataclass
class Config:
    # ---------------- Simulation control ----------------
    num_users: int = 20              # 사용자 수 K
    num_timeslots: int = 7           # 전체 시뮬레이션 타임 슬롯 수
    seed: Optional[int] = 0          # 랜덤 시드 (None이면 고정 X)
    objective: str = "rate"          # "rate" 또는 "pf"

    T: float = 4.848e-3              # 슬롯 전송 시간 ΔT [초]

    bandwidth: float = 80e6          # 전체 대역폭 80 MHz
    rb_bandwidth: float = 2.16e6     # 26-tone RB 대역폭 (2.16 MHz)

    num_antennas: int = 4            # AP 안테나 수 N

    tx_power_dbm: float = 23.0       # 송신 전력(dBm)
    tx_power_watt: float = field(init=False)      # 송신 전력(W)

    noise_dbm_hz: float = -174.0     # N0 [dBm/Hz]
    N0: float = field(init=False)                     # 열잡음 파워 스펙트럼 밀도(W/Hz)

    noise_power_per_rb: float = field(init=False)     # σ² = N0 × B_RB

    pathloss_gamma: float = 3.8      # 경로손실 지수 γ


    dist_min: float = 10.0           # 최소 거리
    dist_max: float = 100.0          # 최대 거리

    # Queue model
    queue_min: float = 0.0           # 초기 큐 최소값
    queue_max: float = 50.0          # 초기 큐 최대값

    arrival_min: float = 0.0         # 슬롯당 도착 패킷 최소값
    arrival_max: float = 10.0        # 슬롯당 도착 패킷 최대값

    # layers
    num_layers: int = 3
    layer0_rb: int = 37              # 26-tone RB 개수
    layer1_rb: int = 16              # 52-tone RB 개수(근사)
    layer2_rb: int = 8               # 106-tone RB 개수(근사)

    # RB 개수 리스트 [L0, L1, L2] = 각 레이어별 RB 개수 리스트    
    layer_rb_counts: List[int] = field(init=False)

    # RB mapping (layer → layer 0 index mapping)
    # (ℓ, i) → base RB indices (layer 0 기준 index) 매핑
    # layer_to_base_rb[ℓ][i] = [base_idx0, base_idx1, ...]
    layer_to_base_rb: List[List[List[int]]] = field(init=False)


    def __post_init__(self):

        # 랜덤 시드 고정
        if self.seed is not None:
            np.random.seed(self.seed)

        # dBm → Watt 변환
        self.tx_power_watt = 10 ** ((self.tx_power_dbm - 30) / 10)

        # N0(dBm/Hz) → W/Hz 변환
        self.N0 = 10 ** ((self.noise_dbm_hz - 30) / 10)

        # 한 RB의 잡음 전력 σ² = N0 × B_RB
        self.noise_power_per_rb = self.N0 * self.rb_bandwidth

        # 각 레이어의 RB 개수
        self.layer_rb_counts = [self.layer0_rb, self.layer1_rb, self.layer2_rb]


        # RB mapping
        q = self.layer0_rb
        layer0_map = [[i] for i in range(q)]
        
        layer1_map = [
            [0, 1],   # RB1_0
            [2, 3],   # RB1_1
            [5, 6],   # RB1_2
            [7, 8],   # RB1_3
            [9, 10], # RB1_4
            [11, 12], # RB1_5
            [14, 15], # RB1_6
            [16, 17], # RB1_7
            [19, 20], # RB1_8
            [21, 22], # RB1_9
            [24, 25], # RB1_10
            [26, 27], # RB1_11
            [28, 29], # RB1_12
            [30, 31], # RB1_13
            [33, 34], # RB1_14
            [35, 36], # RB1_15
        ]
        
        layer2_map = [
            [0, 1, 2, 3],   # RB2_0
            [5, 6, 7, 8],   # RB2_1
            [9, 10, 11, 12],   # RB2_2
            [14, 15, 16, 17],   # RB2_3
            [19, 20, 21, 22], # RB2_4
            [24, 25, 26, 27], # RB2_5
            [28, 29, 30, 31], # RB2_6
            [33, 34, 35, 36], # RB2_7
        ]
        
        self.layer_to_base_rb = [layer0_map, layer1_map, layer2_map]
