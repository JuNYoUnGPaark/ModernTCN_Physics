# Physics-guided deep learning (2020)

## 1. Core Idea: “물리학을 딥러닝에 통합해보자”

- 이를 물리 기반 딥러닝 (Physics-Guided DL. 이하 PGDL)이라고 하며 다음과 같은 한계를 보완한다.
    - *Data-Scarce*: 물리 법칙이 데이터의 빈틈을 메워준다
    - *Generalizability*: 학습 데이터가 없던 상황에서도 물리적으로 타당한 예측을 돕는다
    - *Interpretability*: 왜 그런 예측을 했는지 보다 쉽게 설명 가능하다

## 2. Categories

- **category1: 물리 정보 손실 (Physics-Informed Loss)**
    - PINN이 해당
    - `Total_loss= CE_loss + λ * Physics_loss`
    - 어떤 DL model에도 쉽게 적용 가능
    
- **category2: 물리 제약 아키텍쳐 (Physics-Constrained Architecture)**
    - 모델 구조 자체에 물리법칙을 ‘하드 코딩’하여 물리 법칙을 위반하는 예측을 근본적으로 할 수 없도록 설계하는 방식
    - 에너지 보존, 중력 가속도, 질량 보존 등
    - 카테고리1보다 훨씬 강력하게 물리 법칙을 보장

- **category3: 하이브리드 모델 (Hybrid Physics-DL Models)**
    - 물리 기반의 전통적인 시뮬레이션 모델 + DL 모델을 결합
    - 이미 잘 작동하는 기존의 물리 모델을 활용하면서 DL로 약점 보완 가능

## **3. Category2: 물리 제약 아키텍쳐 (Physics-Constrained Architecture)**

- `acc_pred, gyro_pred`를 예측
    
    → `total_acc = acc + gravity`인 점을 이용
    
    → `gravity_pred = total_acc - body_acc_pred`를 계산 (예측이 아님!)
    

---

*Q. 계산된 `gravity_pred`가 물리적으로 타당한지 어떻게 판단할까?* 

*A1. 중력가속도는 평균적으로 $9.8m/s^2$이므로 Magnitude 비교?*

*⇒ 물리적으로 가장 명확하지만 데이터를 보통 정규화하기 때문에 비정규화 과정을 커져 `gravity_pred` 를 계산하고 이를 비교해야하기 때문에 복잡하다.*

*A2. 중력의 특성 (부드러움 Smoothness) 이용이 타당*

*⇒ 물리적 특성 (1) `body_acc` : 매우 거칠고 빠르게 변한다. (2) 반면, `gravity` : 어떤 동작을 해도 매우 부드럽게(Smoothly) 변한다.* 

---

- **새로운 손실함수 정의**: `Physics_loss = smoothness`
- **작동 방식**
    1. `gravity_pred = total_acc - body_acc_pred` 로 계산
    2. `gravity_pred` 벡터의 시간 축 변화율(Jerk)을 계산
    
    ```python
    # pred_gravity의 시간 축 변화율(Jerk) 계산
    # (B, T, 3) -> (B, T-1, 3)
    gravity_jerk = torch.diff(pred_gravity, n=1, dim=1)
    ```
    
    1. Jerk가 0에 가깝도록 손실 함수가 강제한다. 
    
    ```python
    # 변화율(Jerk)이 0이 되도록 강제 (Jerk의 L2 Norm/MSE)
    loss_phys_residual = torch.mean(gravity_jerk**2)
    ```
    

```python
# --- 카테고리 2 방식의 손실 계산 ---

# 1. 모델에서 6채널 예측값 받기
# physics_pred shape: (B, T, 6)
logits, physics_pred = model(X, return_physics=True)
pred_body_acc = physics_pred[:, :, 0:3]
pred_gyro = physics_pred[:, :, 3:6]

# 2. X_raw에서 실제값 가져오기
true_body_acc = X_raw[:, :, 0:3]
true_gyro = X_raw[:, :, 3:6]
true_total_acc = X_raw[:, :, 6:9]

# 3. [L_recon] 데이터 복원 손실 (기존과 동일)
loss_recon = F.smooth_l1_loss(pred_body_acc, true_body_acc) + \
             F.smooth_l1_loss(pred_gyro, true_gyro)

# 4. [L_residual] 물리 잔차 손실 (Smoothness)
#    물리 법칙(total = body + gravity)을 '아키텍처'로 활용
pred_gravity = true_total_acc - pred_body_acc

#    pred_gravity의 시간 축 변화율(Jerk) 계산
gravity_jerk = torch.diff(pred_gravity, n=1, dim=1)

#    변화율이 0이 되도록 강제 (Jerk의 L2 Norm 최소화)
loss_phys_residual = torch.mean(gravity_jerk**2)

# 5. [L_total] 최종 손실
loss_ce = F.cross_entropy(logits, y)
loss_phys_total = loss_recon + lambda_residual * loss_phys_residual
loss = loss_ce + lambda_phys * loss_phys_total
```

- **작동 방식 상세**

**가정**: 

- Epoch1: 모델이 `body_acc_pred` 를 예측
- Loss 계산: `Physics_loss()` 함수가 `Physics_loss = 1.3` 을 계산
- Total Loss: `Total_loss = CE_loss + λ_phys * (L_recon + λ_resid * 1.3)`
    
    
    1. 손실 발생 및 역전파 시작
    2. Jerk 추적 → 손실값의 원인 찾기
    3. Grvity 추적 → `gravity_pred` 가 시간에 따라 변했기 때문에 loss가 발생한 것
    4. Body Acc가 원인 제공자: `gravity_pred = total_acc - body_acc_pred` 로 계산되기 때문
    5. 모델의 가중치 수정 
    
    해당 과정 반복…
    

---

*Q. Total_acc가 “body_acc + gravity”으로 완벽하게 깔끔한 합이 아니라면 이와 같이 ‘완벽한 부드러운 중력 신호’라는 가정으로 코드를 구성해도 될까?*

*A. UCI-HAR의 `body_acc` = HighPassFiler(`true_total_acc`)로 만들어졌다.* 

- *필터링 과정에서 미세한 신호 지연이나 왜곡이 발생했을 수 있고,*
- *엄밀한 강제성으로 인해서 유용한 신호가 아니라 분류를 방해하는 노이즈로 작용될 수 있다.*
- *따라서, 이것이 PINN과 같은 카테고리1 방식이 더 선호될 수 있는 이유. 우선 실험 후 더 강건하게 작동하는 방식을 선택해야한다.*

*Q. 앞으로 비교할 4가지 데이터의 엄밀함 순서는? (여기 기록용)*

*A. (MHEALTH ≈ PAMAP2) > UCI-HAR > WISDM*

---

## 4. Notice

- 해당 방식은 UCI-HAR과 같이 total_acc 원본과 필터를 적용하여 gravity를 제거한 body_acc를 둘 다 제공하는 특이한 케이스의 데이터만 가능 (향후 나머지 3개 데이터셋과는 비교 불가)

## 5. 실험결과

- 기존 `Best Test F1` = 0.9729
- `lambda_phys_residual`: 카테고리2 적용 비율
- `lambda_phys`: 전체 물리함수 적용 비율
- PINN 실험결과를 고려해 `lambda_phys=0.1` 로 우선 고정 후 내부 변수부터 실험 (Inside-Out으로)

| **`lambda_phys_residual`**  | **F1-score** |
| --- | --- |
| 0.1 | 0.9748 |
| **0.2** | **0.9769**  |
| 0.3 | 0.9762  |
| 0.4 | 0.9763  |
| 0.5 | 0.9757  |
| 1.0 | 0.9728  |
| 1.5 | 0.9713  |
| 2.0 | 0.9692  |
| 5.0 | X |
- `lambda_phys_residual = 0.2` 가 최적값이므로 이제 전체 물리손실 사용비율 실험

| **`lambda_phys`** | **F1-score** |
| --- | --- |
| 0.05 | 0.9722  |
| 0.1 | **0.9769**  |
| 0.15 | 0.9724  |
| 0.2 | 0.9711  |
| 0.25 | 0.9725  |


<img width="989" height="590" alt="image" src="https://github.com/user-attachments/assets/d8052e58-e78a-4eca-b8b3-bad78f4ca609" />
<img width="989" height="590" alt="image" src="https://github.com/user-attachments/assets/e465d8e9-755d-4bf9-936c-1b7ccd654fab" />
