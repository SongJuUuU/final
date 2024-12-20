import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import optuna
from datetime import datetime
import json
import os

# GPU 사용 가능 여부 확인
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# 결과 저장을 위한 디렉토리 생성
RESULTS_DIR = 'training_results'
os.makedirs(RESULTS_DIR, exist_ok=True)

# 1. 데이터 전처리
def prepare_data():
    # 데이터 로드
    batting = pd.read_csv('Batting.csv')
    awards = pd.read_csv('AwardsPlayers.csv')
    
    # MVP 수상 데이터만 필터링
    mvp_awards = awards[awards['awardID'] == 'Most Valuable Player']
    
    # 연도별 수상자 목록 생성
    mvp_winners = set(zip(mvp_awards['playerID'], mvp_awards['yearID']))
    
    # 타격 지표에 MVP 수상 여부 추가
    batting['MVP'] = batting.apply(lambda x: 1 if (x['playerID'], x['yearID']) in mvp_winners else 0, axis=1)
    
    # 필요한 특성 선택
    features = ['G', 'AB', 'R', 'H', '2B', '3B', 'HR', 'RBI', 'SB', 'BB', 'SO']
    
    # 결측치 처리
    batting[features] = batting[features].fillna(0)
    
    # 타율(AVG) 계산 추가
    batting['AVG'] = batting['H'] / batting['AB']
    batting['AVG'] = batting['AVG'].fillna(0)
    
    features.append('AVG')
    
    return batting[features].values, batting['MVP'].values

# 2. PyTorch 모델 정의
class MVPPredictor(nn.Module):
    def __init__(self, input_size, hidden_sizes, dropout_rate):
        super(MVPPredictor, self).__init__()
        
        layers = []
        prev_size = input_size
        
        # 동적으로 히든 레이어 생성
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size
        
        # 출력 레이어
        layers.append(nn.Linear(prev_size, 1))
        layers.append(nn.Sigmoid())
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

# 3. 학습 함수
def train_model(X_train, y_train, X_val, y_val, model, criterion, optimizer, num_epochs, patience=10):
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.FloatTensor(y_train).to(device)
    X_val_tensor = torch.FloatTensor(X_val).to(device)
    y_val_tensor = torch.FloatTensor(y_val).to(device)
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_epoch = 0
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': []
    }
    
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor.view(-1, 1))
        
        loss.backward()
        optimizer.step()
        
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor.view(-1, 1))
            val_predictions = (val_outputs > 0.5).float()
            val_accuracy = (val_predictions == y_val_tensor.view(-1, 1)).float().mean()
        
        history['train_loss'].append(loss.item())
        history['val_loss'].append(val_loss.item())
        history['val_accuracy'].append(val_accuracy.item())
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch}')
            break
            
    return history, best_val_loss.item(), best_epoch

def objective(trial):
    # 하이퍼파라미터 공간 정의
    hidden_sizes = [
        trial.suggest_int(f'hidden_size_1', 16, 256),
        trial.suggest_int(f'hidden_size_2', 16, 128)
    ]
    params = {
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-2),
        'dropout_rate': trial.suggest_uniform('dropout_rate', 0.1, 0.5),
        'batch_norm': trial.suggest_categorical('batch_norm', [True, False]),
        'num_epochs': trial.suggest_int('num_epochs', 50, 200),
        'optimizer_name': trial.suggest_categorical('optimizer_name', ['Adam', 'RMSprop', 'SGD'])
    }
    
    # 데이터 준비
    X, y = prepare_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 모델 초기화
    model = MVPPredictor(
        input_size=X_train.shape[1],
        hidden_sizes=hidden_sizes,
        dropout_rate=params['dropout_rate']
    ).to(device)
    
    # 옵티마이저 설정
    optimizers = {
        'Adam': torch.optim.Adam,
        'RMSprop': torch.optim.RMSprop,
        'SGD': torch.optim.SGD
    }
    optimizer = optimizers[params['optimizer_name']](model.parameters(), lr=params['learning_rate'])
    
    # 손실 함수
    criterion = nn.BCELoss()
    
    # 모델 학습
    history, best_val_loss, best_epoch = train_model(
        X_train_scaled, y_train,
        X_test_scaled, y_test,
        model, criterion, optimizer,
        params['num_epochs']
    )
    
    # 결과 저장
    trial_results = {
        'params': params,
        'hidden_sizes': hidden_sizes,
        'best_val_loss': best_val_loss,
        'best_epoch': best_epoch,
        'history': history
    }
    
    # 결과를 JSON 파일로 저장
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'{RESULTS_DIR}/trial_{trial.number}_{timestamp}.json'
    with open(filename, 'w') as f:
        json.dump(trial_results, f, indent=4)
    
    return best_val_loss

def main():
    print("Starting hyperparameter optimization...")
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50)  # 50회 시도
    
    print("\nBest trial:")
    trial = study.best_trial
    
    print("Value: ", trial.value)
    print("Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    
    # 최적의 하이퍼파라미터로 최종 모델 학습
    print("\nTraining final model with best parameters...")
    
    # 최종 결과 저장
    final_results = {
        'best_params': trial.params,
        'best_value': trial.value,
        'optimization_history': [
            {
                'trial_number': t.number,
                'value': t.value,
                'params': t.params
            }
            for t in study.trials
        ]
    }
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    with open(f'{RESULTS_DIR}/final_results_{timestamp}.json', 'w') as f:
        json.dump(final_results, f, indent=4)

if __name__ == "__main__":
    main() 