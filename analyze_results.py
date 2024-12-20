import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_training_results(results_dir='training_results'):
    # 모든 trial 결과 파일 읽기
    trial_files = [f for f in os.listdir(results_dir) if f.startswith('trial_')]
    
    # 결과 저장을 위한 리스트
    results_data = []
    
    for file in trial_files:
        with open(os.path.join(results_dir, file), 'r') as f:
            data = json.load(f)
            
            # 주요 메트릭 추출
            trial_result = {
                'trial': file,
                'val_loss': data['best_val_loss'],
                'best_epoch': data['best_epoch'],
                'learning_rate': data['params']['learning_rate'],
                'dropout_rate': data['params']['dropout_rate'],
                'optimizer': data['params']['optimizer_name'],
                'hidden_size_1': data['hidden_sizes'][0],
                'hidden_size_2': data['hidden_sizes'][1],
                'final_val_accuracy': data['history']['val_accuracy'][-1]
            }
            results_data.append(trial_result)
    
    # DataFrame 생성
    results_df = pd.DataFrame(results_data)
    
    # 결과 정렬 (검증 손실 기준)
    results_df = results_df.sort_values('val_loss')
    
    print("\n=== 성능 분석 ===")
    print(f"총 시도 횟수: {len(results_df)}")
    print("\n최고 성능 모델:")
    best_model = results_df.iloc[0]
    print(f"검증 손실: {best_model['val_loss']:.4f}")
    print(f"검증 정확도: {best_model['final_val_accuracy']:.4f}")
    print("\n최적 하이퍼파라미터:")
    print(f"Learning Rate: {best_model['learning_rate']}")
    print(f"Dropout Rate: {best_model['dropout_rate']}")
    print(f"Optimizer: {best_model['optimizer']}")
    print(f"Hidden Sizes: [{best_model['hidden_size_1']}, {best_model['hidden_size_2']}]")
    
    # 성능 분포 시각화
    plt.figure(figsize=(15, 10))
    
    # 1. 검증 손실 분포
    plt.subplot(2, 2, 1)
    sns.histplot(results_df['val_loss'])
    plt.title('Validation Loss Distribution')
    
    # 2. 정확도 분포
    plt.subplot(2, 2, 2)
    sns.histplot(results_df['final_val_accuracy'])
    plt.title('Validation Accuracy Distribution')
    
    # 3. Learning Rate vs Loss
    plt.subplot(2, 2, 3)
    plt.scatter(results_df['learning_rate'], results_df['val_loss'])
    plt.xscale('log')
    plt.xlabel('Learning Rate')
    plt.ylabel('Validation Loss')
    plt.title('Learning Rate vs Loss')
    
    # 4. Dropout Rate vs Loss
    plt.subplot(2, 2, 4)
    plt.scatter(results_df['dropout_rate'], results_df['val_loss'])
    plt.xlabel('Dropout Rate')
    plt.ylabel('Validation Loss')
    plt.title('Dropout Rate vs Loss')
    
    plt.tight_layout()
    plt.savefig('training_analysis.png')
    plt.close()
    
    # 옵티마이저별 성능 비교
    print("\n옵티마이저별 평균 성능:")
    print(results_df.groupby('optimizer')['val_loss'].agg(['mean', 'std', 'count']))
    
    return results_df

if __name__ == "__main__":
    results_df = analyze_training_results() 