import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from collections import Counter
import copy
import os
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from models.fedbn_model import FedBNModelSimple

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 7
NUM_FEATURES = 15

UNIFIED_CLASSES = {
    0: 'Benign', 1: 'DoS_DDoS', 2: 'Probe_Recon', 3: 'Brute_Force',
    4: 'Web_Attack', 5: 'Botnet_Malware', 6: 'Spoofing_MITM'
}

# Client to dataset mapping
CLIENT_CONFIG = {
    0: {'slice': 'eMBB-1', 'dataset': 'cicids2017', 'partition': 0},
    1: {'slice': 'eMBB-2', 'dataset': 'cicids2017', 'partition': 1},
    2: {'slice': 'URLLC-1', 'dataset': 'nslkdd', 'partition': 0},
    3: {'slice': 'URLLC-2', 'dataset': 'nslkdd', 'partition': 1},
    4: {'slice': 'mMTC', 'dataset': 'ciciot2023', 'partition': 0},
}

MAX_SAMPLES_PER_CLIENT = 50000
TASK1_CLASSES = [0, 1, 2]  # Common across all clients



# Data Loading


def load_client_data(client_id, data_dir='data/processed_unified'):
    # Load data for a specific client.
    config = CLIENT_CONFIG[client_id]
    dataset = config['dataset']
    partition = config['partition']
    
    X_train = np.load(f'{data_dir}/{dataset}_X_train.npy')
    y_train = np.load(f'{data_dir}/{dataset}_y_train.npy')
    X_test = np.load(f'{data_dir}/{dataset}_X_test.npy')
    y_test = np.load(f'{data_dir}/{dataset}_y_test.npy')
    
    # Partition data if multiple clients share a dataset
    dataset_clients = [cid for cid, c in CLIENT_CONFIG.items() if c['dataset'] == dataset]
    num_clients_for_dataset = len(dataset_clients)
    
    if num_clients_for_dataset > 1:
        total = len(X_train)
        per_client = total // num_clients_for_dataset
        start = partition * per_client
        end = start + per_client if partition < num_clients_for_dataset - 1 else total
        X_train = X_train[start:end]
        y_train = y_train[start:end]
    
    # Subsample if too large
    if len(X_train) > MAX_SAMPLES_PER_CLIENT:
        np.random.seed(42 + client_id)
        indices = np.random.choice(len(X_train), MAX_SAMPLES_PER_CLIENT, replace=False)
        X_train = X_train[indices]
        y_train = y_train[indices]
    
    return X_train, y_train, X_test, y_test


def create_tasks(X_train, y_train, X_test, y_test):
    # Create Task 1 and Task 2 splits
    task1_train_mask = np.isin(y_train, TASK1_CLASSES)
    task1_test_mask = np.isin(y_test, TASK1_CLASSES)
    
    unique_classes = sorted(list(set(y_train.tolist())))
    task2_classes = [c for c in unique_classes if c not in TASK1_CLASSES]
    
    if len(task2_classes) == 0:
        # This client only has Task 1 classes - split some off for Task 2
        task2_classes = [TASK1_CLASSES[-1]]
    
    task2_train_mask = np.isin(y_train, task2_classes)
    task2_test_mask = np.isin(y_test, task2_classes)
    
    X_task2 = X_train[task2_train_mask]
    y_task2 = y_train[task2_train_mask]
    
    # Oversample if tiny
    if 0 < len(X_task2) < 1000:
        repeat_factor = max(1, 1000 // len(X_task2))
        X_task2 = np.tile(X_task2, (repeat_factor, 1))
        y_task2 = np.tile(y_task2, repeat_factor)
    
    return {
        'task1': {
            'train': (X_train[task1_train_mask], y_train[task1_train_mask]),
            'test': (X_test[task1_test_mask], y_test[task1_test_mask]),
            'classes': TASK1_CLASSES
        },
        'task2': {
            'train': (X_task2, y_task2),
            'test': (X_test[task2_test_mask], y_test[task2_test_mask]),
            'classes': task2_classes
        },
        'full_test': (X_test, y_test)
    }



# Continual Learning State 


class ContinualLearningState:
    # Stores EWC Fisher matrix and Replay buffer per client. Persists across rounds.
    
    def __init__(self, client_id):
        self.client_id = client_id
        self.fisher_dict = {}  # Task ID -> Fisher matrix
        self.optimal_params = {}  # Task ID -> params after task
        self.replay_X = None
        self.replay_y = None


# Global state - one per client
CLIENT_STATES = {cid: ContinualLearningState(cid) for cid in CLIENT_CONFIG.keys()}



# Training Functions


def train_client(model, X, y, strategy, task_id, client_id, epochs=2, ewc_lambda=1000, batch_size=256):
    # Train a client model using the specified CL strategy
    state = CLIENT_STATES[client_id]
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    dataset = TensorDataset(torch.FloatTensor(X), torch.LongTensor(y))
    
    # Handle small datasets
    actual_batch_size = min(batch_size, max(len(X) // 4, 2))
    train_loader = DataLoader(dataset, batch_size=actual_batch_size, shuffle=True, drop_last=(len(X) > actual_batch_size))
    
    if len(train_loader) == 0:
        train_loader = DataLoader(dataset, batch_size=len(X), shuffle=True)
    
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for data, target in train_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            
            # Base loss
            if strategy in ['replay', 'ewc_replay'] and state.replay_X is not None and task_id > 1:
                # Mix with replay buffer
                n_replay = min(data.size(0), len(state.replay_X))
                replay_indices = np.random.choice(len(state.replay_X), n_replay, replace=False)
                replay_X = torch.FloatTensor(state.replay_X[replay_indices]).to(DEVICE)
                replay_y = torch.LongTensor(state.replay_y[replay_indices]).to(DEVICE)
                
                combined_X = torch.cat([data, replay_X], dim=0)
                combined_y = torch.cat([target, replay_y], dim=0)
                
                output = model(combined_X)
                loss = criterion(output, combined_y)
            else:
                output = model(data)
                loss = criterion(output, target)
            
            # EWC penalty
            if strategy in ['ewc', 'ewc_replay'] and len(state.fisher_dict) > 0:
                ewc_loss = 0
                for prev_task, fisher in state.fisher_dict.items():
                    for n, p in model.named_parameters():
                        if n in fisher:
                            ewc_loss += (fisher[n] * (p - state.optimal_params[prev_task][n]) ** 2).sum()
                loss = loss + ewc_lambda * ewc_loss
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Track accuracy on current data only
            with torch.no_grad():
                output_eval = model(data)
                _, predicted = output_eval.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        if total > 0 and epoch == epochs - 1:
            acc = 100. * correct / total
            print(f"    Client {client_id} Epoch {epoch+1}: Loss={total_loss/len(train_loader):.4f}, Acc={acc:.2f}%")
    
    return model


def compute_fisher(model, X, y, batch_size=256):
    # Compute Fisher Information Matrix
    dataset = TensorDataset(torch.FloatTensor(X), torch.LongTensor(y))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    criterion = nn.CrossEntropyLoss()
    
    model.eval()
    fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
    num_samples = 0
    
    for data, target in loader:
        data, target = data.to(DEVICE), target.to(DEVICE)
        model.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        
        for n, p in model.named_parameters():
            if p.grad is not None:
                fisher[n] += p.grad.data ** 2
        num_samples += data.size(0)
    
    for n in fisher:
        fisher[n] /= num_samples
    
    return fisher


def update_replay_buffer(X, y, buffer_size=2000):
    # Create a class-balanced replay buffer
    unique_classes = np.unique(y)
    samples_per_class = buffer_size // len(unique_classes)
    
    buffer_X, buffer_y = [], []
    for cls in unique_classes:
        cls_indices = np.where(y == cls)[0]
        n = min(samples_per_class, len(cls_indices))
        selected = np.random.choice(cls_indices, n, replace=False)
        buffer_X.append(X[selected])
        buffer_y.append(y[selected])
    
    return np.concatenate(buffer_X, axis=0), np.concatenate(buffer_y, axis=0)


def evaluate_model(model, X, y, batch_size=256):
    # Evaluate model on a dataset.
    if len(X) == 0:
        return 0.0, 0.0
    
    dataset = TensorDataset(torch.FloatTensor(X), torch.LongTensor(y))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    criterion = nn.CrossEntropyLoss()
    
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            total_loss += criterion(output, target).item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    loss = total_loss / len(loader) if len(loader) > 0 else 0
    acc = correct / total if total > 0 else 0
    return loss, acc



# Federated Averaging 


def fedavg_aggregate(client_models, client_weights=None):
    """Aggregate client models using FedAvg. Returns averaged state dict.
    
    FedBN: Skip BatchNorm running stats.
    """
    if client_weights is None:
        client_weights = [1.0 / len(client_models)] * len(client_models)
    
    # Normalize weights
    total = sum(client_weights)
    client_weights = [w / total for w in client_weights]
    
    # Get first model's state dict as template
    avg_state = copy.deepcopy(client_models[0].state_dict())
    
    for key in avg_state.keys():
        # FedBN skip BN statistics (keep local)
        if 'running_mean' in key or 'running_var' in key or 'num_batches_tracked' in key:
            continue
        
        # Aggregate
        avg_state[key] = torch.zeros_like(avg_state[key], dtype=torch.float32)
        for i, client_model in enumerate(client_models):
            avg_state[key] += client_weights[i] * client_model.state_dict()[key].float()
        
        # Restore original dtype
        avg_state[key] = avg_state[key].to(client_models[0].state_dict()[key].dtype)
    
    return avg_state


def load_avg_into_clients(client_models, avg_state):
    # Load averaged weights into all client models (keeping local BN stats)
    for model in client_models:
        current_state = model.state_dict()
        for key, value in avg_state.items():
            # Skip BN stats (they stay local)
            if 'running_mean' in key or 'running_var' in key or 'num_batches_tracked' in key:
                continue
            current_state[key] = value
        model.load_state_dict(current_state, strict=False)



# Main Experiment


def run_experiment(strategy_type='naive', num_rounds_task1=10, num_rounds_task2=3, epochs_per_round=2):
    # Run complete federated continual learning experiment
    
    print("\n" + "="*70)
    print(f"STRATEGY: {strategy_type.upper()}")
    print("="*70)
    
    # Reset global state for this experiment
    global CLIENT_STATES
    CLIENT_STATES = {cid: ContinualLearningState(cid) for cid in CLIENT_CONFIG.keys()}
    
    # Load data and create tasks for each client
    print("\nLoading data for all clients...")
    client_tasks = {}
    for cid in CLIENT_CONFIG.keys():
        X_train, y_train, X_test, y_test = load_client_data(cid)
        client_tasks[cid] = create_tasks(X_train, y_train, X_test, y_test)
        
        config = CLIENT_CONFIG[cid]
        t1_size = len(client_tasks[cid]['task1']['train'][0])
        t2_size = len(client_tasks[cid]['task2']['train'][0])
        print(f"  Client {cid} ({config['slice']}/{config['dataset']}): "
              f"Task1={t1_size:,}, Task2={t2_size:,}")
    
    # Create client models
    client_models = {cid: FedBNModelSimple(input_dim=NUM_FEATURES, num_classes=NUM_CLASSES).to(DEVICE) 
                     for cid in CLIENT_CONFIG.keys()}
    
    
    # TASK 1: Train on initial attacks
    
    print(f"\n{'='*70}")
    print(f"TASK 1 TRAINING ({num_rounds_task1} rounds)")
    print("="*70)
    
    for round_num in range(num_rounds_task1):
        print(f"\nRound {round_num + 1}/{num_rounds_task1}")
        
        # Each client trains locally
        for cid in CLIENT_CONFIG.keys():
            X, y = client_tasks[cid]['task1']['train']
            if len(X) > 0:
                client_models[cid] = train_client(
                    client_models[cid], X, y, 
                    strategy=strategy_type, task_id=1, client_id=cid,
                    epochs=epochs_per_round
                )
        
        # Federated averaging
        client_weights = [len(client_tasks[cid]['task1']['train'][0]) for cid in CLIENT_CONFIG.keys()]
        avg_state = fedavg_aggregate(list(client_models.values()), client_weights)
        load_avg_into_clients(list(client_models.values()), avg_state)
    
    # Evaluate Task 1 accuracy (across all clients)
    print("\n--- Task 1 Evaluation (after training) ---")
    task1_accs = []
    task1_sizes = []
    for cid in CLIENT_CONFIG.keys():
        X_test, y_test = client_tasks[cid]['task1']['test']
        if len(X_test) > 0:
            _, acc = evaluate_model(client_models[cid], X_test, y_test)
            task1_accs.append(acc * len(X_test))
            task1_sizes.append(len(X_test))
            print(f"  Client {cid} ({CLIENT_CONFIG[cid]['slice']}): {acc*100:.2f}% ({len(X_test):,} samples)")
    
    task1_final_acc = sum(task1_accs) / sum(task1_sizes) * 100 if task1_sizes else 0
    print(f"\n✓ Task 1 Global Accuracy: {task1_final_acc:.2f}%")
    
    
    # SAVE STATE FOR CL STRATEGIES
    
    print(f"\n{'='*70}")
    print("SAVING CL STATE (EWC Fisher & Replay Buffer)")
    print("="*70)
    
    for cid in CLIENT_CONFIG.keys():
        X, y = client_tasks[cid]['task1']['train']
        if len(X) == 0:
            continue
        
        state = CLIENT_STATES[cid]
        
        # Compute Fisher for EWC strategies
        if strategy_type in ['ewc', 'ewc_replay']:
            print(f"  Computing Fisher for Client {cid}...")
            state.fisher_dict[1] = compute_fisher(client_models[cid], X, y)
            state.optimal_params[1] = {n: p.data.clone() for n, p in client_models[cid].named_parameters()}
        
        # Update replay buffer for Replay strategies
        if strategy_type in ['replay', 'ewc_replay']:
            print(f"  Updating replay buffer for Client {cid}...")
            state.replay_X, state.replay_y = update_replay_buffer(X, y)
    
    
    # TASK 2: Train on new attacks
    
    print(f"\n{'='*70}")
    print(f"TASK 2 TRAINING ({num_rounds_task2} rounds)")
    print("="*70)
    
    for round_num in range(num_rounds_task2):
        print(f"\nRound {round_num + 1}/{num_rounds_task2}")
        
        for cid in CLIENT_CONFIG.keys():
            X, y = client_tasks[cid]['task2']['train']
            if len(X) > 0:
                client_models[cid] = train_client(
                    client_models[cid], X, y, 
                    strategy=strategy_type, task_id=2, client_id=cid,
                    epochs=epochs_per_round
                )
        
        # Federated averaging
        client_weights = [len(client_tasks[cid]['task2']['train'][0]) for cid in CLIENT_CONFIG.keys()]
        total_weight = sum(client_weights)
        if total_weight > 0:
            avg_state = fedavg_aggregate(list(client_models.values()), client_weights)
            load_avg_into_clients(list(client_models.values()), avg_state)
    
    # Evaluate Task 2 accuracy
    print("\n--- Task 2 Evaluation (after training) ---")
    task2_accs = []
    task2_sizes = []
    for cid in CLIENT_CONFIG.keys():
        X_test, y_test = client_tasks[cid]['task2']['test']
        if len(X_test) > 0:
            _, acc = evaluate_model(client_models[cid], X_test, y_test)
            task2_accs.append(acc * len(X_test))
            task2_sizes.append(len(X_test))
            print(f"  Client {cid} ({CLIENT_CONFIG[cid]['slice']}): {acc*100:.2f}% ({len(X_test):,} samples)")
    
    task2_final_acc = sum(task2_accs) / sum(task2_sizes) * 100 if task2_sizes else 0
    print(f"\n✓ Task 2 Global Accuracy: {task2_final_acc:.2f}%")
    
    
    # FULL EVALUATION (measures forgetting)
    
    print(f"\n{'='*70}")
    print("FULL EVALUATION (all classes - measures forgetting)")
    print("="*70)
    
    full_accs = []
    full_sizes = []
    task1_retention = []  # How well Task 1 is retained
    
    for cid in CLIENT_CONFIG.keys():
        # Full test
        X_test, y_test = client_tasks[cid]['full_test']
        if len(X_test) > 0:
            _, acc = evaluate_model(client_models[cid], X_test, y_test)
            full_accs.append(acc * len(X_test))
            full_sizes.append(len(X_test))
            
            # Task 1 retention
            X_t1, y_t1 = client_tasks[cid]['task1']['test']
            if len(X_t1) > 0:
                _, t1_acc = evaluate_model(client_models[cid], X_t1, y_t1)
                task1_retention.append((cid, t1_acc))
                print(f"  Client {cid} ({CLIENT_CONFIG[cid]['slice']}): "
                      f"Full={acc*100:.2f}%, Task1 Retention={t1_acc*100:.2f}%")
    
    full_final_acc = sum(full_accs) / sum(full_sizes) * 100 if full_sizes else 0
    avg_task1_retention = sum(acc for _, acc in task1_retention) / len(task1_retention) * 100 if task1_retention else 0
    
    # True forgetting = Task 1 accuracy BEFORE Task 2 training vs AFTER
    true_forgetting = task1_final_acc - avg_task1_retention
    
    print(f"\n✓ Full Accuracy: {full_final_acc:.2f}%")
    print(f"✓ Task 1 Retention: {avg_task1_retention:.2f}% (was {task1_final_acc:.2f}%)")
    print(f"✓ True Forgetting: {true_forgetting:.2f}%")
    
    return {
        'strategy': strategy_type,
        'task1_accuracy': task1_final_acc,
        'task2_accuracy': task2_final_acc,
        'full_accuracy': full_final_acc,
        'task1_retention': avg_task1_retention,
        'forgetting': true_forgetting
    }


if __name__ == '__main__':
    print("\n" + "="*70)
    print("MULTI-DATASET FEDERATED CONTINUAL LEARNING FOR 6G (FIXED)")
    print("="*70)
    print("\nClient Configuration:")
    for cid, config in CLIENT_CONFIG.items():
        print(f"  Client {cid}: {config['slice']:<10s} → {config['dataset']}")
    
    print(f"\nTask Design:")
    print(f"  Task 1 Classes: {TASK1_CLASSES} (Benign, DoS_DDoS, Probe_Recon)")
    print(f"  Task 2 Classes: Remaining classes per client")
    
    results = {}
    
    for strategy in ['naive', 'ewc', 'replay', 'ewc_replay']:
        results[strategy] = run_experiment(
            strategy_type=strategy,
            num_rounds_task1=10,
            num_rounds_task2=3,
            epochs_per_round=2
        )
        print("\n" + "="*70)
        print(f"✓ {strategy.upper()} EXPERIMENT COMPLETE")
        print("="*70 + "\n")
    
    # Save results
    os.makedirs('results', exist_ok=True)
    with open('results/multidataset_fcl_final_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Display final comparison
    print("\n" + "="*70)
    print("FINAL RESULTS COMPARISON")
    print("="*70)
    print(f"{'Strategy':<15} {'Task1':<10} {'Task2':<10} {'Full':<10} {'Retention':<12} {'Forgetting':<12}")
    print("-"*75)
    for strategy, res in results.items():
        print(f"{strategy.upper():<15} "
              f"{res['task1_accuracy']:>8.2f}% "
              f"{res['task2_accuracy']:>8.2f}% "
              f"{res['full_accuracy']:>8.2f}% "
              f"{res['task1_retention']:>10.2f}% "
              f"{res['forgetting']:>10.2f}%")
    
    print("\n✓ Results saved to: results/multidataset_fcl_final_results.json")
    
    # Show improvement from naive
    print("\n" + "="*70)
    print("CL STRATEGY IMPROVEMENTS OVER NAIVE")
    print("="*70)
    naive_forgetting = results['naive']['forgetting']
    for strategy, res in results.items():
        if strategy != 'naive':
            improvement = naive_forgetting - res['forgetting']
            print(f"  {strategy.upper():<15} Forgetting: {res['forgetting']:>6.2f}% "
                  f"(↓ {improvement:.2f}% vs Naive)")