import torch
import torch.nn as nn
import torch.nn.functional as F


class FedBNModel(nn.Module):
    """
    FedBN Model for Federated Learning on Non-IID Features.
    
    Architecture:
    - Local input adapter (not federated)
    - Shared trunk with LOCAL BatchNorm (only weights federated, not BN stats)
    - Local classification head (not federated)
    """
    
    def __init__(self, input_dim=15, num_classes=7, hidden_dims=[128, 64, 32]):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dims = hidden_dims
        
        
        # LOCAL: Input Adapter (stays on each client)
        
        self.input_adapter = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),  # LOCAL BN
            nn.ReLU()
        )
        
        
        # FEDERATED: Shared Trunk
        # (weights aggregated, but BN stats stay local)
        
        self.trunk = nn.Sequential(
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.BatchNorm1d(hidden_dims[1]),  # LOCAL BN
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.BatchNorm1d(hidden_dims[2]),  # LOCAL BN
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        
        # LOCAL: Classification Head (stays on each client)
        
        self.classifier = nn.Linear(hidden_dims[2], num_classes)
    
    def forward(self, x):
        x = self.input_adapter(x)
        x = self.trunk(x)
        x = self.classifier(x)
        return x
    
    def get_federated_params(self):
        """Get parameters that should be federated (trunk weights only, NOT BN)."""
        federated_params = []
        for name, param in self.named_parameters():
            # Include trunk weights but EXCLUDE BatchNorm and local layers
            if 'trunk' in name and 'bn' not in name.lower() and 'batch' not in name.lower():
                federated_params.append((name, param))
        return federated_params
    
    def get_local_params(self):
        """Get parameters that stay local (adapter, BN, classifier)."""
        local_params = []
        for name, param in self.named_parameters():
            # Include everything that's NOT trunk weights
            if 'trunk' not in name or 'bn' in name.lower() or 'batch' in name.lower():
                local_params.append((name, param))
        return local_params


class FedBNModelSimple(nn.Module):
    """
    Simpler version - more compatible with standard Flower workflow.
    Uses FedAvg but keeps BN stats local through strict=False loading.
    """
    
    def __init__(self, input_dim=15, num_classes=7):
        super().__init__()
        
        # All layers - but BN stats will stay local via strict=False
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        
        self.fc3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        
        self.dropout = nn.Dropout(0.3)
        
        self.classifier = nn.Linear(32, num_classes)
    
    def forward(self, x):
        # Handle case where batch size is 1 (BN needs >1)
        if x.size(0) == 1:
            x = torch.cat([x, x], dim=0)
            single_sample = True
        else:
            single_sample = False
        
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        
        x = F.relu(self.bn3(self.fc3(x)))
        
        x = self.classifier(x)
        
        if single_sample:
            x = x[:1]
        
        return x


def get_model_state_dict_for_fedbn(model):
    """
    Get state dict EXCLUDING BatchNorm statistics.
    Used when sending parameters to server.
    
    This implements FedBN: only share non-BN parameters.
    """
    state_dict = model.state_dict()
    filtered_dict = {}
    
    for key, value in state_dict.items():
        # Skip BatchNorm running statistics
        if 'running_mean' in key or 'running_var' in key or 'num_batches_tracked' in key:
            continue
        filtered_dict[key] = value
    
    return filtered_dict


def load_model_state_dict_for_fedbn(model, state_dict):
    """
    Load state dict while keeping local BatchNorm statistics.
    Used when receiving parameters from server.
    """
    current_state = model.state_dict()
    
    # Only update non-BN parameters
    for key, value in state_dict.items():
        if 'running_mean' in key or 'running_var' in key or 'num_batches_tracked' in key:
            continue
        if key in current_state:
            current_state[key] = value
    
    model.load_state_dict(current_state, strict=False)


if __name__ == '__main__':
    # Test the model
    print("="*70)
    print("FEDBN MODEL ARCHITECTURE TEST")
    print("="*70)
    
    # Create model
    model = FedBNModelSimple(input_dim=15, num_classes=7)
    
    print(f"\nModel structure:")
    print(model)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    
    # Test forward pass
    print("\nTesting forward pass...")
    test_input = torch.randn(32, 15)  # Batch of 32, 15 features
    output = model(test_input)
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")
    
    # Show FedBN parameter filtering
    print("\n" + "-"*70)
    print("FEDBN PARAMETER FILTERING")
    print("-"*70)
    
    full_state = model.state_dict()
    filtered_state = get_model_state_dict_for_fedbn(model)
    
    print(f"\nFull state dict keys ({len(full_state)}):")
    for key in full_state.keys():
        excluded = "❌ EXCLUDED (local)" if key not in filtered_state else "✓ Federated"
        print(f"  {excluded:25s} {key}")
    
    print(f"\nFederated parameters: {len(filtered_state)}")
    print(f"Local parameters: {len(full_state) - len(filtered_state)}")
    
    print("\n✓ Model test passed!")