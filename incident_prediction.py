import numpy as np
import xgboost as xgb
from sklearn.metrics import classification_report, precision_recall_curve
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.calibration import CalibratedClassifierCV
import warnings

warnings.filterwarnings('ignore')

def generate_synthetic_data(n_steps=10000, seed=42):
    """
    Generuje syntetyczne dane szeregów czasowych (CPU, Memory) oraz wstrzykuje awarie.
    """
    np.random.seed(seed)
    time = np.arange(n_steps)
    
    # Bazowe metryki
    metric_cpu = np.sin(time * 0.1) + np.random.normal(0, 0.5, n_steps)
    metric_mem = time * 0.0001 + (metric_cpu * 0.3) + np.random.normal(0, 0.2, n_steps)
    incidents = np.zeros(n_steps)

    # Wstrzykiwanie 150 losowych incydentów
    incident_indices = np.random.choice(n_steps - 20, size=150, replace=False)
    for idx in incident_indices:
        if np.random.rand() > 0.5:
            # Nagły skok
            metric_cpu[idx:idx+3] += 5.0  
            metric_mem[idx:idx+3] += 3.0
            incidents[idx:idx+3] = 1
        else:
            # Stopniowa degradacja (memory leak)
            metric_mem[idx:idx+10] += np.linspace(0, 4.0, 10)
            incidents[idx:idx+10] = 1
            
    return metric_cpu, metric_mem, incidents

def create_features_and_targets(metric_cpu, metric_mem, incidents, W=20, H=5):
    """
    Przekształca surowe metryki w okna przesuwne (sliding windows) i generuje cechy inżynieryjne.
    """
    X_list, y_list = [], []
    x_axis = np.arange(W) 

    for i in range(W, len(metric_cpu) - H):
        window_cpu = metric_cpu[i-W:i]
        window_mem = metric_mem[i-W:i]
        
        # Obliczanie podstawowych statystyk
        cpu_mean, cpu_std = np.mean(window_cpu), np.std(window_cpu)
        mem_mean, mem_std = np.mean(window_mem), np.std(window_mem)
        
        # Kwantyle i anomalie bieżące
        cpu_q25, cpu_q75 = np.percentile(window_cpu, 25), np.percentile(window_cpu, 75)
        mem_q25, mem_q75 = np.percentile(window_mem, 25), np.percentile(window_mem, 75)
        
        cpu_current_diff = window_cpu[-1] - cpu_mean
        mem_current_diff = window_mem[-1] - mem_mean
        
        # Łączenie w jeden wektor cech
        features = np.concatenate([
            window_cpu, window_mem,
            [cpu_mean, cpu_std, np.polyfit(x_axis, window_cpu, 1)[0], cpu_q25, cpu_q75, cpu_current_diff],
            [mem_mean, mem_std, np.polyfit(x_axis, window_mem, 1)[0], mem_q25, mem_q75, mem_current_diff],
            [cpu_mean * mem_mean, cpu_mean / (mem_mean + 1e-5)]
        ])
        
        # Target: 1 jeśli awaria wystąpi w oknie H
        target = 1 if np.any(incidents[i:i+H]) else 0
        X_list.append(features)
        y_list.append(target)

    return np.array(X_list), np.array(y_list)

def optimize_business_threshold(y_true, y_probs, cost_fp=1, cost_fn=5):
    """
    Szuka optymalnego progu klasyfikacji minimalizującego całkowity koszt biznesowy.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)
    
    best_cost = float('inf')
    optimal_threshold = 0.5

    for threshold in thresholds:
        y_pred_temp = (y_probs >= threshold).astype(int)
        FP = np.sum((y_pred_temp == 1) & (y_true == 0))
        FN = np.sum((y_pred_temp == 0) & (y_true == 1))
        
        total_cost = (FP * cost_fp) + (FN * cost_fn)
        if total_cost < best_cost:
            best_cost = total_cost
            optimal_threshold = threshold
            
    return optimal_threshold

def main():
    print("1. Generowanie danych syntetycznych...")
    cpu, mem, incidents = generate_synthetic_data()
    
    print("2. Inżynieria cech (Sliding Window W=20, Horizon H=5)...")
    X, y = create_features_and_targets(cpu, mem, incidents)
    
    # Podział 70/30 (Bez szuflowania - Time Series!)
    train_end = int(len(X) * 0.7)
    X_train, y_train = X[:train_end], y[:train_end]
    X_test, y_test = X[train_end:], y[train_end:]

    scale_pos_weight = (len(y_train) - np.sum(y_train)) / np.sum(y_train)

    print("3. Trening modelu (Grid Search & TimeSeriesSplit)...")
    tscv = TimeSeriesSplit(n_splits=3)
    param_grid = {'max_depth': [4, 6], 'learning_rate': [0.05, 0.1], 'n_estimators': [100, 200]}
    
    base_model = xgb.XGBClassifier(
        scale_pos_weight=scale_pos_weight, 
        max_delta_step=1, 
        random_state=42, 
        eval_metric='logloss'
    )
    
    grid_search = GridSearchCV(estimator=base_model, param_grid=param_grid, cv=tscv, scoring='f1', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    print(f"   Najlepsze parametry: {grid_search.best_params_}")

    print("4. Kalibracja prawdopodobieństwa...")
    calibrated_model = CalibratedClassifierCV(grid_search.best_estimator_, method='sigmoid', cv=tscv)
    calibrated_model.fit(X_train, y_train)

    print("5. Optymalizacja progu na podstawie kosztów (Cost-Sensitive Learning)...")
    train_probs = calibrated_model.predict_proba(X_train)[:, 1]
    optimal_threshold = optimize_business_threshold(y_train, train_probs, cost_fp=1, cost_fn=5)
    print(f"   Optymalny próg: {optimal_threshold:.4f}")

    print("\n=== RAPORT KOŃCOWY (ZBIÓR TESTOWY) ===")
    test_probs = calibrated_model.predict_proba(X_test)[:, 1]
    final_predictions = (test_probs >= optimal_threshold).astype(int)
    print(classification_report(y_test, final_predictions))

if __name__ == "__main__":
    main()