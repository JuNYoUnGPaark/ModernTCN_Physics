# Physics Loss 관련 논문 정리

## 1. “고전적” 물리 정보 신경망 (PINN)

- 핵심 아이디어: `Total_Loss = Loss_data + Loss_physics`
    - `Loss_data` : y(Target)과 y_pred의 차이 (현재 코드: `loss_ce`)
    - `Loss_physics` : y_pred를 미분하여 물리방정식 (ex: $f(\hat y, \frac {d\hat y}{dt})=0$)에 넣었을 때 그 결과가 0으로부터 얼마나 벗어나는지 ⇒ ‘Pure Physics Loss 형태’

***논문 1)"Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations" (Raissi, Perdikaris, & Karniadakis, 2019)***

- PINN 분야를 개척한 논문(필독). 딥러닝으로 어떻게 미분방정식을 풀고, 데이터 부족 영역을 물리 법칙으로 채워 넣을 수 있는지에 대한 idea 제공

**논문 2) *"Physics-guided deep learning: A review of recent advances and applications" (Willard et al., 2020/2022)***

- PINN을 포함하여 물리를 딥러닝에 통합하는 다양한 방법론 (loss function, architecture, hybrid)을 총 정리한 논문.

## 2. IMU 센서의 물리적 특성 활용

- UCI-HAR 데이터는 IMU 센서 데이터. 이 센서들(`acc`, `gyro`) 사이의 물리적 관계를 직접 활용하는 방법에 대한 논문

***논문1) "OrientNet: Robust IMU-based Orientation Estimation with Deep* Learning" (Yi et al., 2020)**

- IMU 센서로 방향, 자세를 추정하는 것은 `total_acc` 에서 `gravity` 벡터를 정확히 분리하는 것이 핵심. 딥러닝 모델이 “중력 분해”라는 물리적 task를 어떻게 수행하는지 설명

***논문2) "RIDI: Robust IMU Double Integration" (Yan et al., 2018)***

- 가속도계를 이중적분하여 위치를 추적하는 것은 IMU의 고전적인 물리 문제. 딥러닝을 사용하여 이 과정을 어떠헥 보정하는지 다루며, 센서의 미/적분 관계(`Jerk Loss`)를 손실함수에 사용하는 Idea 제공.

## 3. 신호처리 & SSL 활용

→ 여기서 더 넘어가면 Physics_Loss Upgrade와는 멀어지니 우선 skip…
