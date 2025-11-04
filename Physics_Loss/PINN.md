# Physics-Informed Neural Networks (2019)

### 1. The Core Idea

$$
Total \ Loss = L_{data} + L_{physics}
$$

- **$L_{data}$ (데이터 손실)**
    - 실제 관측데이터와 신경망의 예측값의 차를 측정
    - 전통적인 DL의 SL Loss(MSE 등)과 동일
- **$L_{physics}$ (물리 손실)**
    - PINN의 핵심
    - 신경망의 예측값이 물리법칙(→ 편미분 방정식, PDE 등)을 얼마나 잘 따르는지를 측정
    - Ex)어떤 물리법칙이 $f(u)=0$ 라면, $L_{physics}$는 신경망의 예측값을 방정식에 대입한 값의 제곱의 평균이 된다.
    - 실제 정답 데이터가 없는 지점에서도 계산이 가능하다.
        
        → 물리 손실은 미리 정의한 물리법칙(물리방정식)에 대한 잔차값만을 계산하기 때문! 
        

### 2. How it Works

- PINN은 물리손실을 계산하기 위해 `자동 미분`을 핵심 기술로 사용한다.
    1. 신경망 $u_{NN}(x, t)$를 정의한다. 
    2. 손실 $L_{physics}$를 계산하려면 $du/dt,\ du/dx$와 같은 미분 항이 필요하다. 
    3. Pytorch의 `torch.autograd.grad`를 이용하면 신경망의 출력값으로부터 입력에 대한 도함수를 정확하게 계산할 수 있다. 
    4. 이렇게 얻은 미분 값들을 물리 방정식에 대입하여 물리손실을 계산하고 데이터 손실(CE Loss등)과 더해 전체 손실을 구성한 뒤 Backprop.을 수행한다. 

### 3. PINN 방식의 업그레이드 방안

- PINN의 $L_{physics}$는 예측값들 사이의 관계가 물리 법칙을 만족해야한다는 것.
- “중력 분해”를 PINN의 물리손실 관점에서 적용 가능.
    1. 물리법칙 정의
        
        $f = (pred\ total\_acc) - (pred\ body\_acc+pred\ gravity)$
        
    2. 새로운 물리손실 정의
        
        $loss_{physics}=torch.mean(f^2)$
        
- 모델이 $f$를 0으로 만들도록 강제.
- 최종 손실: $L=L_{ce}+\lambda_1*L_{recon}+\lambda_2*L_{physics}$(여기서 $L_{recon}$은 기존의 복원 예측)

---

- *Code 수정 사항 정리*

```python
class PhysicsModernTCNHAR(BaseModernTCNHAR):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        hidden_dim = self.head.in_features
        
        # [수정] self.physics_head의 마지막 레이어 출력을 6 -> 9로 변경
        self.physics_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            # 6채널 (body_acc, gyro) -> 9채널 (body_acc, gyro, gravity)
            nn.Linear(hidden_dim // 2, 9) 
        )
```

```python
def physics_loss_pinn_v1(physics_pred, X_raw, lambda_residual=1.0):
    """
    PINN 철학을 적용한 물리 손실 함수.
    
    물리 법칙: total_acc = body_acc + gravity
    잔차 (f): (pred_body_acc + pred_gravity) - true_total_acc
    
    L_total = L_recon (데이터 손실) + lambda_residual * L_phys_residual (물리 손실)
    """
    
    # --- 1. 예측값 분리 (물리 헤드 출력 = 9채널) ---
    # (B, T, 9) -> (B, T, 3) 3개
    pred_body_acc = physics_pred[:, :, 0:3]
    pred_gyro     = physics_pred[:, :, 3:6]
    pred_gravity  = physics_pred[:, :, 6:9] # 예측된 중력 (Latent)

    # --- 2. 실제값 분리 (원본 X = 9채널) ---
    true_body_acc = X_raw[:, :, 0:3]
    true_gyro     = X_raw[:, :, 3:6]
    true_total_acc = X_raw[:, :, 6:9] # 실제 총 가속도

    # --- 3. 손실 계산 ---
    
    # [A] L_recon (데이터 손실): 기존의 복원 손실 (body_acc, gyro만)
    loss_recon = F.smooth_l1_loss(pred_body_acc, true_body_acc) + \
                 F.smooth_l1_loss(pred_gyro, true_gyro)

    # [B] L_phys_residual (물리 손실): PINN의 잔차 손실
    
    # 부품으로 예측한 총 가속도
    pred_total_from_parts = pred_body_acc + pred_gravity 
    
    # 잔차: (예측 부품 합)과 (실제 총 가속도)가 일치해야 함
    loss_phys_residual = F.smooth_l1_loss(pred_total_from_parts, true_total_acc)

    # --- 4. 최종 손실 조합 ---
    # L_recon (2개 항의 합)과 L_phys_residual (1개 항)을 가중치(lambda)로 조절
    return loss_recon + lambda_residual * loss_phys_residual
```

```python
def train_physics(model, train_loader, test_loader, device, epochs=50, 
                  lambda_phys=0.05, 
                  lambda_phys_residual=1.0, # [추가] L_recon과 L_residual 비율
                  log_every=1):
                  
    (생략)
    
		logits, physics = model(X, return_physics=True)
		loss_ce = F.cross_entropy(logits, y)
		
		# [수정] 새로운 물리 손실 함수 호출
		loss_phys = physics_loss_pinn_v1(physics, X, lambda_phys_residual)
		# loss_phys는 이제 L_recon + L_residual의 합임.
		# lambda_phys는 CE와 (전체 물리손실) 간의 비율을 조절.
		loss = loss_ce + lambda_phys * loss_phys
		loss.backward()
		
		(생략)
```

```python
# [수정] train_physics 호출 시 'lambda_phys_residual' 인자 추가
    f1_3, acc_3, hist = train_physics(model3, train_loader, test_loader, device,
                                      epochs=EPOCHS, 
                                      lambda_phys=0.05,       # (기존) CE와 Phys의 비율
                                      lambda_phys_residual=1.0, # (신규) Phys 내부의 잔차 손실 비율
                                      log_every=25)
```

---

### 4. Notice

- `lambda` 로 표현되는 손실 가중치 Hyperparameter는 PINN 계열과 같이 Multi-task 학습에서 성능에 가장 큰 영향을 미치는 가장 민감한 요소
    - `lamda_phys` (현재=0.05): 이 값이 너무 작으면 쓴 보람이 없고 너무 크면 본 task인 분류 성능이 떨어진다.
    - `lambda_phys_residual` (현재=1.0): 이 값이 너무 작으면 PINN idea가 무시되고 너무 크면 `total = body + gravity` 라는 관계만 맞추려고 한다.
- 현재: 가장 좋은 조합으로 Tunning 완료됨.

### 5. Results

- 기존 `Best Test F1` = 0.9729

| `lambda_phys_residual` | F1-Score |
| --- | --- |
| 0.1 | 0.9719 |
| 0.5 | 0.9723 |
| 1.0 | 0.9731  |
| 1.5 | 0.9727 |
| 2.0 | 0.9744 |
| **2.5** | **0.9772** |
| 3.0 | 0.9763 |
| 3.5 | 0.9763 |
| **4.0** | **0.9772** |
| 4.5 | 0.9771 |
| 5.0 | 0.9749 |
- `lambda_phys_residual` = 2.5
    
    
    | `lamda_phys` | F1-Score |
    | --- | --- |
    | 0.01 | 0.9706 |
    | 0.05 | 0.9772 |
    | **0.1** | **0.9783** |
    | 0.15 | 0.9777 |
    | 0.2 | 0.9757 |
- `lambda_phys_residual` = 4.0

	| `lamda_phys` | F1-Score |
	| --- | --- |
	| 0.01 | 0.9734 |
	| 0.05 | 0.9772 |
	| 0.1 | 0.9765 |
	| 0.15 | 0.9770 |
	| 0.2 | 0.9750 |

<img width="989" height="590" alt="image" src="https://github.com/user-attachments/assets/187d9477-3e18-4d25-95d3-2e91ae664216" />

<img width="989" height="590" alt="image" src="https://github.com/user-attachments/assets/ca55fd89-ce70-4b16-9e40-63527689dd9c" />

<img width="989" height="590" alt="image" src="https://github.com/user-attachments/assets/0b03a501-630c-4544-9efe-652bdb9f7a37" />
