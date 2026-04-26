# ============================================================================
# TEMPORAL POOLING SNN - GRAYSCALE ENCODING (PAPER'S APPROACH)
# ============================================================================
# Implementation following Zhang et al. (arXiv:2312.12899v3)
# Uses Strategy 1: Grayscale-Equivalent encoding
#

#
# KEY FEATURES:
# - RGB → Grayscale conversion (0.299*R + 0.587*G + 0.114*B)
# - Single pass through 784-neuron reservoir
# - 15,680 features (784 neurons × 20 temporal bins)
# - Matches paper's architecture exactly
# - Ridge Regression for readout (paper's validated method)
# ============================================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import json
from tqdm import tqdm
import sys
import time
import datetime
import platform
from collections import defaultdict
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.linear_model import RidgeClassifier  # For Ridge Regression
import pickle


ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR / "src"))
from model import Circuit2D

print("="*70)
print("TEMPORAL POOLING SNN - GRAYSCALE ENCODING (PAPER'S APPROACH)")
print("="*70)
print("✓ GPU Optimized")
print("✓ 500ns temporal windows")
print("✓ 20 temporal bins")
print("✓ TRUE multi-task learning (independent tasks)")
print("✓ Strategy 1: RGB → Grayscale encoding (matches paper!)")
print("✓ 784 neurons, 15,680 features")
print("✓ Ridge Regression readout (paper's validated method)")
print("✓ COMPREHENSIVE data saving enabled")
print("="*70)

# ============================================================================
# DATA LOGGER CLASS (UNCHANGED)
# ============================================================================

class ComprehensiveLogger:
    """
    Logs and saves EVERYTHING for publication and analysis.
    """
    
    def __init__(self, save_dir='./output/'):
        self.save_dir = save_dir
        self.start_time = time.time()
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Storage for all metrics
        self.data = {
            'metadata': self._collect_system_info(),
            'configuration': {},
            'training': {
                'epochs': [],
                'batch_metrics': [],
                'learning_curves': defaultdict(list)
            },
            'evaluation': {
                'confusion_matrices': {},
                'per_class_accuracy': {},
                'classification_reports': {}
            },
            'reservoir_dynamics': {
                'sample_currents': [],
                'sample_temperatures': [],
                'pooled_features': []
            },
            'model_checkpoints': [],
            'timing': {}
        }
        
        print(f"\n✓ Logger initialized: {self.timestamp}")
        print(f"  Save directory: {save_dir}")
    
    def _collect_system_info(self):
        """Collect system and environment information."""
        return {
            'timestamp': self.timestamp,
            'python_version': platform.python_version(),
            'pytorch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
            'device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
            'num_gpus': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'platform': platform.platform(),
            'processor': platform.processor()
        }
    
    def log_configuration(self, config):
        """Log model and training configuration."""
        self.data['configuration'] = config
        print(f"\n✓ Configuration logged")
    
    def log_epoch(self, epoch, metrics, duration):
        """Log epoch-level metrics."""
        epoch_data = {
            'epoch': epoch,
            'duration_seconds': duration,
            'metrics': metrics,
            'timestamp': time.time() - self.start_time
        }
        self.data['training']['epochs'].append(epoch_data)
    
    def log_batch(self, epoch, batch_idx, task, loss, accuracy):
        """Log batch-level metrics."""
        batch_data = {
            'epoch': epoch,
            'batch': batch_idx,
            'task': task,
            'loss': float(loss),
            'accuracy': float(accuracy)
        }
        self.data['training']['batch_metrics'].append(batch_data)
    
    def log_confusion_matrix(self, task, y_true, y_pred, class_names=None):
        """Log confusion matrix for a task."""
        cm = confusion_matrix(y_true, y_pred)
        
        self.data['evaluation']['confusion_matrices'][task] = {
            'matrix': cm.tolist(),
            'class_names': class_names
        }
        
        # Per-class accuracy
        per_class_acc = cm.diagonal() / cm.sum(axis=1)
        self.data['evaluation']['per_class_accuracy'][task] = {
            f'class_{i}': float(acc) for i, acc in enumerate(per_class_acc)
        }
        
        # Classification report
        if class_names:
            report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
            self.data['evaluation']['classification_reports'][task] = report
    
    def log_reservoir_sample(self, image, current_trajectory, pooled_features, label, task='digit'):
        """Log reservoir dynamics for a sample."""
        sample_data = {
            'label': int(label),
            'task': task,
            'current_shape': current_trajectory.shape,
            'pooled_shape': pooled_features.shape,
            # Save statistics instead of full arrays to save space
            'current_stats': {
                'mean': float(current_trajectory.mean()),
                'std': float(current_trajectory.std()),
                'min': float(current_trajectory.min()),
                'max': float(current_trajectory.max())
            },
            'pooled_stats': {
                'mean': float(pooled_features.mean()),
                'std': float(pooled_features.std()),
                'min': float(pooled_features.min()),
                'max': float(pooled_features.max())
            }
        }
        
        self.data['reservoir_dynamics']['sample_currents'].append(sample_data)
        
        # Save a few full samples separately
        if len(self.data['reservoir_dynamics']['sample_currents']) <= 10:
            np.save(
                f"{self.save_dir}/sample_{len(self.data['reservoir_dynamics']['sample_currents'])}_current.npy",
                current_trajectory
            )
            np.save(
                f"{self.save_dir}/sample_{len(self.data['reservoir_dynamics']['sample_currents'])}_pooled.npy",
                pooled_features
            )
    
    def log_model_checkpoint(self, epoch, model_path, metrics):
        """Log model checkpoint information."""
        checkpoint_data = {
            'epoch': epoch,
            'path': model_path,
            'metrics': metrics,
            'timestamp': time.time() - self.start_time
        }
        self.data['model_checkpoints'].append(checkpoint_data)
    
    def save_all(self):
        """Save all collected data to files."""
        print("\n" + "="*70)
        print("SAVING COMPREHENSIVE DATA")
        print("="*70)
        
        # Add final timing
        self.data['timing']['total_duration_seconds'] = time.time() - self.start_time
        self.data['timing']['total_duration_readable'] = str(datetime.timedelta(seconds=int(time.time() - self.start_time)))
        
        # Save main JSON
        json_path = f"{self.save_dir}/comprehensive_results_{self.timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(self.data, f, indent=2, default=str)
        print(f"✓ Main JSON saved: {json_path}")
        
        # Save training curves separately (for easy plotting)
        curves_path = f"{self.save_dir}/training_curves_{self.timestamp}.json"
        curves_data = {
            'epochs': [e['epoch'] for e in self.data['training']['epochs']],
            'metrics': {
                task: [e['metrics'][task] for e in self.data['training']['epochs']]
                for task in ['digit', 'color', 'parity']
            },
            'durations': [e['duration_seconds'] for e in self.data['training']['epochs']]
        }
        with open(curves_path, 'w') as f:
            json.dump(curves_data, f, indent=2)
        print(f"✓ Training curves saved: {curves_path}")
        
        # Save confusion matrices as numpy arrays
        for task, cm_data in self.data['evaluation']['confusion_matrices'].items():
            cm_path = f"{self.save_dir}/confusion_matrix_{task}_{self.timestamp}.npy"
            np.save(cm_path, np.array(cm_data['matrix']))
            print(f"✓ Confusion matrix ({task}) saved: {cm_path}")
        
        # Save batch-level data separately (can be large)
        batch_path = f"{self.save_dir}/batch_metrics_{self.timestamp}.pkl"
        with open(batch_path, 'wb') as f:
            pickle.dump(self.data['training']['batch_metrics'], f)
        print(f"✓ Batch metrics saved: {batch_path}")
        
        # Save summary
        summary_path = f"{self.save_dir}/SUMMARY_{self.timestamp}.txt"
        self._save_summary(summary_path)
        print(f"✓ Summary saved: {summary_path}")
        
        print("="*70)
        print("✅ ALL DATA SAVED!")
        print("="*70)
    
    def _save_summary(self, path):
        """Save human-readable summary."""
        with open(path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("RGB TEMPORAL POOLING RESERVOIR - RESULTS SUMMARY\n")
            f.write("="*70 + "\n\n")
            
            # System info
            f.write("SYSTEM INFORMATION\n")
            f.write("-" * 70 + "\n")
            for key, value in self.data['metadata'].items():
                f.write(f"{key}: {value}\n")
            f.write("\n")
            
            # Configuration
            f.write("CONFIGURATION\n")
            f.write("-" * 70 + "\n")
            for key, value in self.data['configuration'].items():
                f.write(f"{key}: {value}\n")
            f.write("\n")
            
            # Final results
            if self.data['training']['epochs']:
                final = self.data['training']['epochs'][-1]
                f.write("FINAL RESULTS\n")
                f.write("-" * 70 + "\n")
                for task, acc in final['metrics'].items():
                    f.write(f"{task.capitalize()}: {acc*100:.2f}%\n")
                f.write("\n")
            
            # Timing
            f.write("TIMING\n")
            f.write("-" * 70 + "\n")
            f.write(f"Total duration: {self.data['timing']['total_duration_readable']}\n")
            if self.data['training']['epochs']:
                avg_epoch_time = np.mean([e['duration_seconds'] for e in self.data['training']['epochs']])
                f.write(f"Average epoch time: {avg_epoch_time:.2f} seconds\n")
            f.write("\n")
            
            # Files created
            f.write("FILES CREATED\n")
            f.write("-" * 70 + "\n")
            f.write(f"1. comprehensive_results_{self.timestamp}.json\n")
            f.write(f"2. training_curves_{self.timestamp}.json\n")
            f.write(f"3. batch_metrics_{self.timestamp}.pkl\n")
            f.write(f"4. confusion_matrix_*_{self.timestamp}.npy (per task)\n")
            f.write(f"5. sample_*_current.npy (reservoir dynamics)\n")
            f.write(f"6. sample_*_pooled.npy (pooled features)\n")
            f.write(f"7. SUMMARY_{self.timestamp}.txt (this file)\n")

# ============================================================================
# RGB TEMPORAL POOLING RESERVOIR (MODIFIED FOR RGB)
# ============================================================================

class TemporalPoolingReservoir_RGB(nn.Module):
    """
    Temporal pooling reservoir with Grayscale encoding (Strategy 1).
    
    Follows Zhang et al. (arXiv:2312.12899v3) approach:
    - Converts RGB images to grayscale
    - Processes through 784-neuron VO2 reservoir
    - Extracts 15,680 temporal features (784 × 20 bins)
    - Uses Ridge Regression for multi-task readout
    """
    
    def __init__(self, batch_size, logger=None):
        super().__init__()
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = logger
        
        self.batch = batch_size
        self.Nx = 28
        self.Ny = 28
        self.N = 784  # neurons per channel
        
        self.V_min = 10.5  # Paper optimal (was 11.0)
        self.V_max = 12.2  # Paper optimal (was 13.0)
        self.R = 12
        self.noise_strength = 0.0002  # Paper optimal: 0.2 µJ·s^(-1/2) (was 0.001)
        self.Cth_factor = 0.15  # Paper optimal (was 1.0) - CRITICAL for fast spiking!
        self.couple_factor = 0.02
        self.width_factor = 1.0
        self.T_base = 325
        
        self.t_max = 10000
        self.dt = 10
        self.n_step = 1000
        self.window_size = 500
        self.len_y = 50
        self.n_temporal_bins = 20
        
        # Feature dimension: 784 neurons × 20 bins = 15,680 (GRAYSCALE)
        self.feature_dim = self.N * self.n_temporal_bins
        
        print(f"\n{'='*70}")
        print(f"GRAYSCALE ENCODING (PAPER'S APPROACH)")
        print(f"{'='*70}")
        print(f"Device: {self.device}")
        print(f"Neurons: {self.N}")
        print(f"Encoding: RGB → Grayscale → Voltage")
        print(f"Temporal bins: {self.n_temporal_bins}")
        print(f"Feature dim: {self.feature_dim} (matches paper!)")
        print(f"{'='*70}")
        
        # Log configuration
        if self.logger:
            self.logger.log_configuration({
                'architecture': 'Grayscale_Temporal_Pooling',
                'encoding_method': 'Strategy_1_Grayscale_Equivalent',
                'color_conversion': 'RGB → Grayscale (0.299*R + 0.587*G + 0.114*B)',
                'batch_size': batch_size,
                'grid_size': f'{self.Nx}x{self.Ny}',
                'num_neurons': self.N,
                'num_simulations_per_image': 1,
                'V_min': self.V_min,
                'V_max': self.V_max,
                'R': self.R,
                'noise_strength': self.noise_strength,
                'Cth_factor': self.Cth_factor,
                'couple_factor': self.couple_factor,
                'width_factor': self.width_factor,
                'T_base': self.T_base,
                't_max_ns': self.t_max,
                'dt_ns': self.dt,
                'n_steps': self.n_step,
                'window_size_ns': self.window_size,
                'temporal_bins': self.n_temporal_bins,
                'feature_dim': self.feature_dim,
                'device': str(self.device),
                'paper_match': 'Yes - follows Zhang et al. grayscale encoding'
            })
        
        # Single reservoir (will process each channel separately)
        self.reservoir = Circuit2D(
            self.batch, self.Nx, self.Ny,
            self.V_min, self.R,
            self.noise_strength, self.Cth_factor,
            self.couple_factor, self.width_factor,
            self.T_base
        )
        
        if self.device.type == 'cuda':
            self._move_reservoir_to_gpu()
        
        # Task-specific readout heads (now with 3× more features!)
        self.fc_digit = nn.Linear(self.feature_dim, 10)
        self.fc_color = nn.Linear(self.feature_dim, 10)
        self.fc_parity = nn.Linear(self.feature_dim, 2)
        
        print(f"✓ Grayscale reservoir ready with {self.feature_dim:,} features (matches paper!)\n")
    
    def _move_reservoir_to_gpu(self):
        """Move ALL reservoir tensors to GPU."""
        print("  → Moving reservoir to GPU...")
        
        reservoir_attrs = ['V0', 'R0C0', 'C0', 'S_env', 'S_couple', 'Cth', 'T_base', 'V']
        for attr_name in reservoir_attrs:
            if hasattr(self.reservoir, attr_name):
                attr = getattr(self.reservoir, attr_name)
                if torch.is_tensor(attr):
                    setattr(self.reservoir, attr_name, attr.to(self.device))
        
        if hasattr(self.reservoir, 'VO2'):
            vo2 = self.reservoir.VO2
            vo2_attrs = ['R_m', 'R_0', 'E_a', 'T_c', 'w', 'beta', 'gamma',
                         'T_last', 'delta', 'Ea', 'Rm', 'Tc', 'Tr', 'Tpr', 
                         'reversed', 'gr']
            
            for attr_name in vo2_attrs:
                if hasattr(vo2, attr_name):
                    attr = getattr(vo2, attr_name)
                    if torch.is_tensor(attr):
                        setattr(vo2, attr_name, attr.to(self.device))
            
            for attr_name in dir(vo2):
                if not attr_name.startswith('_'):
                    attr = getattr(vo2, attr_name)
                    if torch.is_tensor(attr) and attr.device.type != self.device.type:
                        try:
                            setattr(vo2, attr_name, attr.to(self.device))
                        except:
                            pass
        
        print("  ✓ Reservoir on GPU")
    
    def temporal_pooling_layer(self, current_trajectory):
        """MAX pooling over 500ns windows."""
        batch, N, n_steps = current_trajectory.shape
        trajectory_flat = current_trajectory.reshape(batch * N, 1, n_steps)
        
        pooled = torch.nn.functional.max_pool1d(
            trajectory_flat, kernel_size=self.len_y, stride=self.len_y
        )
        
        return pooled.squeeze(1).reshape(batch, N, self.n_temporal_bins)
    
    def extract_temporal_features(self, rgb_images, log_sample=False, label=None):
        """
        Extract features using GRAYSCALE-EQUIVALENT encoding (Strategy 1).
        
        This follows the paper's approach:
        1. Convert RGB to grayscale using standard luminance formula
        2. Process ONCE through reservoir (784 neurons)
        3. Extract 15,680 features (784 × 20 temporal bins)
        
        Parameters:
        -----------
        rgb_images : torch.Tensor, shape (batch, 3, 28, 28)
            RGB images
        
        Returns:
        --------
        features : torch.Tensor, shape (batch, 15680)
            Temporal features from reservoir
        """
        rgb_images = rgb_images.to(self.device)
        
        # STRATEGY 1: Convert RGB to grayscale (luminance formula)
        # Standard conversion: Gray = 0.299*R + 0.587*G + 0.114*B
        R = rgb_images[:, 0, :, :]  # (batch, 28, 28)
        G = rgb_images[:, 1, :, :]  # (batch, 28, 28)
        B = rgb_images[:, 2, :, :]  # (batch, 28, 28)
        
        grayscale = 0.299 * R + 0.587 * G + 0.114 * B  # (batch, 28, 28)
        grayscale_flat = grayscale.reshape(self.batch, -1)  # (batch, 784)
        
        # Voltage encoding (paper's approach)
        voltages = self.V_min + (self.V_max - self.V_min) * grayscale_flat
        
        # Set reservoir input
        self.reservoir.set_input(V=voltages)
        
        # Initialize state (all neurons at room temperature)
        initial_state = torch.stack([
            torch.zeros(self.batch, self.N, device=self.device),  # Initial current = 0
            torch.ones(self.batch, self.N, device=self.device) * self.T_base  # Initial temp = 325K
        ], dim=1)
        
        # Solve VO2 dynamics
        _, current_trajectory = self.reservoir.solve(initial_state, self.t_max, self.dt)
        
        # Temporal pooling
        pooled_features = self.temporal_pooling_layer(current_trajectory)
        
        # Flatten: (batch, 784, 20) → (batch, 15680)
        features = pooled_features.reshape(self.batch, -1)
        
        # Log sample if requested
        if log_sample and self.logger and label is not None:
            self.logger.log_reservoir_sample(
                rgb_images[0].cpu().numpy(),
                current_trajectory[0].cpu().numpy(),
                pooled_features[0].cpu().numpy(),
                label[0]
            )
        
        return features
    
    def forward(self, rgb_images, task='digit', log_sample=False, label=None):
        """
        Forward pass with RGB images.
        
        Parameters:
        -----------
        rgb_images : torch.Tensor, shape (batch, 3, 28, 28)
            RGB images (NOT grayscale!)
        task : str
            'digit', 'color', or 'parity'
        
        Returns:
        --------
        output : torch.Tensor
            Log probabilities for the specified task
        """
        rgb_images = rgb_images.to(self.device)
        features = self.extract_temporal_features(rgb_images, log_sample, label)
        
        if task == 'digit':
            logits = self.fc_digit(features)
        elif task == 'color':
            logits = self.fc_color(features)
        elif task == 'parity':
            logits = self.fc_parity(features)
        
        return torch.nn.functional.log_softmax(logits, dim=-1)

# ============================================================================
# DATA PREPARATION - RGB PRESERVED (FIXED!)
# ============================================================================

def prepare_colored_mnist_rgb(seed=42):
    """
    Create Colored MNIST dataset.
    
    ENCODING STRATEGY: Grayscale-Equivalent (Strategy 1)
    1. Assign random colors to MNIST digits (for color labels)
    2. Store as RGB images
    3. During training: Convert RGB → Grayscale → Voltage
    
    This preserves the paper's architecture while enabling color classification.
    """
    print("\n" + "="*70)
    print("PREPARING COLORED MNIST DATA (GRAYSCALE ENCODING)")
    print("="*70)
    print("✓ Colors assigned RANDOMLY (independent from digits)")
    print("✓ Stored as RGB images")
    print("✓ Encoding: RGB → Grayscale → Voltage (Strategy 1)")
    print("✓ Matches paper's architecture (784 neurons, 15,680 features)")
    print("="*70)
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    transform = transforms.Compose([transforms.ToTensor()])
    mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    color_palette = torch.tensor([
        [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0],
        [1.0, 0.5, 0.0], [0.5, 0.0, 1.0], [0.0, 0.5, 0.0],
        [0.5, 0.5, 0.5]
    ])
    
    def create_colored_rgb(dataset):
        images, digits, colors, parities = [], [], [], []
        
        for idx in tqdm(range(len(dataset)), desc="Creating RGB colored"):
            image, digit = dataset[idx]
            gray = image.squeeze(0)
            
            color_idx = np.random.randint(0, 10)
            color = color_palette[color_idx]
            
            # Create RGB image by multiplying grayscale by color
            colored = torch.zeros(3, 28, 28)
            for c in range(3):
                colored[c] = gray * color[c]
            
            images.append(colored)
            digits.append(digit)
            colors.append(color_idx)
            parities.append(digit % 2)
        
        return (torch.stack(images), 
                torch.tensor(digits), 
                torch.tensor(colors), 
                torch.tensor(parities))
    
    X_train, y_train_d, y_train_c, y_train_p = create_colored_rgb(mnist_train)
    X_test, y_test_d, y_test_c, y_test_p = create_colored_rgb(mnist_test)
    
    # Data stored as RGB, but will be converted to grayscale during encoding
    print(f"\n✓ Colored MNIST data ready:")
    print(f"  Train: {X_train.shape} (RGB format)")
    print(f"  Test:  {X_test.shape}")
    print(f"\n🎯 Encoding Strategy 1 (Paper's Approach):")
    print(f"   RGB → Grayscale (0.299*R + 0.587*G + 0.114*B)")
    print(f"   Grayscale → Voltage (V_min to V_max)")
    print(f"   Single pass through 784-neuron reservoir")
    print(f"   Result: 15,680 features (matches paper!)")
    print("="*70)
    
    return X_train, y_train_d, y_train_c, y_train_p, X_test, y_test_d, y_test_c, y_test_p

# ============================================================================
# RIDGE REGRESSION TRAINING (PAPER'S APPROACH)
# ============================================================================

def train_with_ridge_regression(model, train_loader, test_loader, device, logger=None, epoch_num=1):
    """
    Train using Ridge Regression instead of Adam optimizer.
    This is the PAPER'S approach - scientifically more rigorous!
    
    Process:
    1. Extract all reservoir features (no training yet)
    2. Train three Ridge Regression classifiers
    3. Evaluate on test set
    
    Returns trained classifiers and results.
    """
    print(f"\n{'='*70}")
    print(f"RIDGE REGRESSION TRAINING (Paper's Approach)")
    print(f"{'='*70}")
    
    # Step 1: Extract ALL training features
    print("Step 1: Extracting reservoir features from training set...")
    model.eval()
    
    all_features = []
    all_labels_digit = []
    all_labels_color = []
    all_labels_parity = []
    
    with torch.no_grad():
        for batch_idx, (batch_x, batch_y_d, batch_y_c, batch_y_p) in enumerate(tqdm(train_loader, desc="Extracting features", leave=False)):
            batch_x = batch_x.to(device)
            
            # Extract reservoir features (RGB processing happens here)
            features = model.extract_temporal_features(batch_x)
            
            all_features.append(features.cpu())
            all_labels_digit.append(batch_y_d)
            all_labels_color.append(batch_y_c)
            all_labels_parity.append(batch_y_p)
    
    # Concatenate all features
    X_train = torch.cat(all_features).numpy()
    y_train_digit = torch.cat(all_labels_digit).numpy()
    y_train_color = torch.cat(all_labels_color).numpy()
    y_train_parity = torch.cat(all_labels_parity).numpy()
    
    print(f"✓ Extracted {X_train.shape[0]} samples with {X_train.shape[1]} features")
    
    # Step 2: Train Ridge Regression classifiers
    print("\nStep 2: Training Ridge Regression classifiers...")
    
    # Task-specific regularization (alpha)
    # More regularization for harder tasks, less for easier tasks
    alpha_digit = 1e-3   # Digit is hardest
    alpha_color = 1e-3   # Color is hard
    alpha_parity = 1e-4  # Parity is easiest (binary classification)
    
    clf_digit = RidgeClassifier(alpha=alpha_digit, max_iter=1000)
    clf_color = RidgeClassifier(alpha=alpha_color, max_iter=1000)
    clf_parity = RidgeClassifier(alpha=alpha_parity, max_iter=1000)
    
    print(f"  Training digit classifier (alpha={alpha_digit})...")
    clf_digit.fit(X_train, y_train_digit)
    
    print(f"  Training color classifier (alpha={alpha_color})...")
    clf_color.fit(X_train, y_train_color)
    
    print(f"  Training parity classifier (alpha={alpha_parity})...")
    clf_parity.fit(X_train, y_train_parity)
    
    print("✓ Ridge Regression training complete!")
    
    # Step 3: Evaluate on test set
    print("\nStep 3: Evaluating on test set...")
    results = evaluate_with_ridge(model, test_loader, clf_digit, clf_color, clf_parity, device, logger)
    
    print(f"\nRESULTS (Epoch {epoch_num}):")
    print(f"  Digit:  {results['digit']*100:.2f}%")
    print(f"  Color:  {results['color']*100:.2f}%")
    print(f"  Parity: {results['parity']*100:.2f}%")
    
    return clf_digit, clf_color, clf_parity, results


def evaluate_with_ridge(model, dataloader, clf_digit, clf_color, clf_parity, device, logger=None):
    """
    Evaluate Ridge Regression classifiers.
    """
    model.eval()
    
    # Extract test features
    all_features = []
    all_labels_digit = []
    all_labels_color = []
    all_labels_parity = []
    
    with torch.no_grad():
        for batch_x, batch_y_d, batch_y_c, batch_y_p in tqdm(dataloader, desc="Evaluating", leave=False):
            batch_x = batch_x.to(device)
            features = model.extract_temporal_features(batch_x)
            
            all_features.append(features.cpu())
            all_labels_digit.append(batch_y_d)
            all_labels_color.append(batch_y_c)
            all_labels_parity.append(batch_y_p)
    
    X_test = torch.cat(all_features).numpy()
    y_test_digit = torch.cat(all_labels_digit).numpy()
    y_test_color = torch.cat(all_labels_color).numpy()
    y_test_parity = torch.cat(all_labels_parity).numpy()
    
    # Get predictions
    pred_digit = clf_digit.predict(X_test)
    pred_color = clf_color.predict(X_test)
    pred_parity = clf_parity.predict(X_test)
    
    # Calculate accuracies
    acc_digit = (pred_digit == y_test_digit).mean()
    acc_color = (pred_color == y_test_color).mean()
    acc_parity = (pred_parity == y_test_parity).mean()
    
    results = {
        'digit': acc_digit,
        'color': acc_color,
        'parity': acc_parity
    }
    
    # Log confusion matrices
    if logger:
        logger.log_confusion_matrix('digit', y_test_digit, pred_digit, 
                                   class_names=[str(i) for i in range(10)])
        logger.log_confusion_matrix('color', y_test_color, pred_color,
                                   class_names=[str(i) for i in range(10)])
        logger.log_confusion_matrix('parity', y_test_parity, pred_parity,
                                   class_names=['even', 'odd'])
    
    return results


# ============================================================================
# RIDGE REGRESSION TRAINING SESSION (PAPER'S APPROACH - REPLACES ADAM)
# ============================================================================

def run_ridge_training_session():
    """
    Complete training using Ridge Regression (Paper's approach).
    
    This replaces the 5-session Adam training with ONE simple session:
    1. Extract all reservoir features (fixed reservoir, no training)
    2. Train Ridge Regression classifiers (fast!)
    3. Evaluate and save results
    
    Why this is better for journal papers:
    - ✅ Paper used this method (96% accuracy)
    - ✅ No backpropagation through reservoir (physically realistic)
    - ✅ Simple linear readout (biologically plausible)
    - ✅ Faster training (minutes instead of hours!)
    - ✅ Standard reservoir computing approach
    """
    
    print("\n" + "="*70)
    print("RIDGE REGRESSION TRAINING (PAPER'S SCIENTIFIC APPROACH)")
    print("="*70)
    print("✓ Replacing Adam optimizer with Ridge Regression")
    print("✓ No epochs needed - trains in ONE pass!")
    print("✓ Scientifically rigorous for journal publication")
    print("="*70)
    
    # Initialize logger
    logger = ComprehensiveLogger(
        log_dir='./logs',
        run_name=f"ridge_regression_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    
    # Prepare data (RGB preserved!)
    print("\n📊 Preparing Colored MNIST dataset (RGB preserved)...")
    X_train, y_train_d, y_train_c, y_train_p, X_test, y_test_d, y_test_c, y_test_p = prepare_colored_mnist_rgb(seed=42)
    
    batch_size = 32
    
    # Set random seeds
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    np.random.seed(42)
    
    train_loader = DataLoader(
        TensorDataset(X_train, y_train_d, y_train_c, y_train_p),
        batch_size=batch_size, shuffle=False, drop_last=True  # No shuffle for Ridge!
    )
    test_loader = DataLoader(
        TensorDataset(X_test, y_test_d, y_test_c, y_test_p),
        batch_size=batch_size, shuffle=False, drop_last=True
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✓ Using device: {device}")
    
    # Initialize model (reservoir only, no trainable readout yet!)
    print("\n" + "="*70)
    print("INITIALIZING RGB RESERVOIR")
    print("="*70)
    model = TemporalPoolingReservoir_RGB(batch_size, logger=logger).to(device)
    
    # Log configuration
    logger.log_configuration({
        'method': 'Ridge Regression (Paper Approach)',
        'optimizer': 'None (Ridge Regression)',
        'reservoir_training': 'Fixed (no backprop)',
        'readout_training': 'Ridge Regression',
        'batch_size': batch_size,
        'train_size': len(X_train),
        'test_size': len(X_test),
        'color_encoding': 'RGB_PRESERVED',
        'V_min': 10.5,
        'V_max': 12.2,
        'Cth_factor': 0.15,
        'noise_strength': 0.0002,
        'alpha_digit': 1e-3,
        'alpha_color': 1e-3,
        'alpha_parity': 1e-4
    })
    
    # Train with Ridge Regression
    print("\n" + "="*70)
    print("TRAINING WITH RIDGE REGRESSION")
    print("="*70)
    
    start_time = time.time()
    clf_digit, clf_color, clf_parity, results = train_with_ridge_regression(
        model, train_loader, test_loader, device, logger, epoch_num=1
    )
    training_time = time.time() - start_time
    
    print(f"\n✓ Training complete in {training_time:.1f} seconds!")
    
    # Save classifiers
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'clf_digit': clf_digit,
        'clf_color': clf_color,
        'clf_parity': clf_parity,
        'results': results,
        'training_time': training_time,
        'config': {
            'V_min': 10.5,
            'V_max': 12.2,
            'Cth_factor': 0.15,
            'noise_strength': 0.0002
        }
    }
    
    checkpoint_path = f"checkpoint_ridge_regression.pth"
    torch.save(checkpoint, checkpoint_path)
    print(f"✓ Saved checkpoint: {checkpoint_path}")
    
    # Final results
    print("\n" + "="*70)
    print("FINAL RESULTS (RIDGE REGRESSION)")
    print("="*70)
    print(f"Digit:  {results['digit']*100:.2f}%")
    print(f"Color:  {results['color']*100:.2f}%")
    print(f"Parity: {results['parity']*100:.2f}%")
    print(f"Training time: {training_time:.1f}s")
    print("="*70)
    
    # Compare to baselines
    print("\nCOMPARISON TO BASELINES:")
    print(f"  Digit:  {results['digit']*100:.2f}% (random: 10%, paper: 96%)")
    print(f"  Color:  {results['color']*100:.2f}% (random: 10%, grayscale: 33%)")
    print(f"  Parity: {results['parity']*100:.2f}% (random: 50%)")
    
    return results, logger, (clf_digit, clf_color, clf_parity)


# ============================================================================
# KEPT FOR COMPATIBILITY (Not used with Ridge Regression)
# ============================================================================

def train_epoch(model, dataloader, optimizer, criterion, device, task, epoch, logger=None):
    """
    LEGACY FUNCTION - Kept for reference but NOT USED with Ridge Regression.
    Ridge Regression doesn't need epoch-based training!
    """
    model.train()
    correct, total, total_loss = 0, 0, 0.0
    batch_count = 0
    
    for batch_idx, (batch_x, batch_y_d, batch_y_c, batch_y_p) in enumerate(tqdm(dataloader, desc=f"Train {task}", leave=False)):
        batch_x = batch_x.to(device)  # RGB images (batch, 3, 28, 28)
        batch_y = (batch_y_d if task == 'digit' else batch_y_c if task == 'color' else batch_y_p).to(device)
        
        # Log reservoir dynamics for first few batches
        log_sample = (logger is not None and batch_idx < 3 and epoch == 1)
        
        optimizer.zero_grad()
        output = model(batch_x, task=task, log_sample=log_sample, label=batch_y)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()
        
        batch_correct = (output.argmax(1) == batch_y).sum().item()
        correct += batch_correct
        total += batch_y.size(0)
        total_loss += loss.item()
        batch_count += 1
        
        # Log batch metrics
        if logger and batch_idx % 100 == 0:  # Log every 100 batches
            logger.log_batch(epoch, batch_idx, task, loss.item(), batch_correct / batch_y.size(0))
    
    return correct / total, total_loss / batch_count

def evaluate_comprehensive(model, dataloader, device, logger=None):
    """Comprehensive evaluation with confusion matrices."""
    model.eval()
    results = {}
    
    # Store predictions for confusion matrices
    all_preds = {task: [] for task in ['digit', 'color', 'parity']}
    all_labels = {task: [] for task in ['digit', 'color', 'parity']}
    
    for task in ['digit', 'color', 'parity']:
        correct, total = 0, 0
        with torch.no_grad():
            for batch_x, batch_y_d, batch_y_c, batch_y_p in dataloader:
                batch_x = batch_x.to(device)  # RGB images
                batch_y = (batch_y_d if task == 'digit' else batch_y_c if task == 'color' else batch_y_p).to(device)
                output = model(batch_x, task=task)
                preds = output.argmax(1)
                
                correct += (preds == batch_y).sum().item()
                total += batch_y.size(0)
                
                # Store for confusion matrix
                all_preds[task].extend(preds.cpu().numpy())
                all_labels[task].extend(batch_y.cpu().numpy())
        
        results[task] = correct / total
        
        # Log confusion matrix
        if logger:
            class_names = [str(i) for i in range(10)] if task in ['digit', 'color'] else ['even', 'odd']
            logger.log_confusion_matrix(task, all_labels[task], all_preds[task], class_names)
    
    return results

# ============================================================================
# VISUALIZATION WITH RGB COMPARISON
# ============================================================================

def plot_results(history, method='RGB MAX', save_path='./output/rgb_pooling_plot.png'):
    """Create training progress plot."""
    epochs = range(1, len(history) + 1)
    
    digit_acc = [h['digit'] * 100 for h in history]
    color_acc = [h['color'] * 100 for h in history]
    parity_acc = [h['parity'] * 100 for h in history]
    
    plt.figure(figsize=(14, 7))
    
    plt.plot(epochs, digit_acc, 'b-o', label='Digit (independent)', linewidth=2.5, markersize=10)
    plt.plot(epochs, color_acc, 'g-s', label='Color (RGB preserved!)', linewidth=2.5, markersize=10)
    plt.plot(epochs, parity_acc, 'r-^', label='Parity', linewidth=2.5, markersize=10)
    plt.axhline(y=90, color='purple', linestyle='--', label='Target (90%)', linewidth=2)
    plt.axhline(y=33.4, color='orange', linestyle=':', label='Grayscale baseline (33%)', linewidth=2)
    plt.axhline(y=10, color='gray', linestyle=':', label='Random (10-class)', linewidth=1.5)
    plt.axhline(y=50, color='gray', linestyle=':', label='Random (2-class)', linewidth=1.5)
    
    plt.xlabel('Epoch', fontsize=14, fontweight='bold')
    plt.ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
    plt.title(f'{method} Pooling - RGB Preserved (Expected: Color >> 33%)', fontsize=16, fontweight='bold')
    plt.legend(fontsize=11, loc='lower right')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.ylim([0, 100])
    plt.xticks(epochs)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved to: {save_path}")
    plt.close()

# ============================================================================
# RIDGE REGRESSION TRAINING (PAPER'S APPROACH)
# ============================================================================

# ============================================================================
# CHECKPOINT MANAGEMENT
# ============================================================================

def save_checkpoint(epoch, model, classifiers, history, logger, save_path):
    """
    Save complete checkpoint including model, ridge classifiers, history, and logger.
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'classifiers': classifiers,  # Ridge classifiers (digit, color, parity)
        'history': history,
        'logger_data': logger.data,
        'timestamp': datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    }
    
    torch.save(checkpoint, save_path)
    print(f"\n✅ CHECKPOINT SAVED: {save_path}")
    print(f"   Epoch: {epoch}, Last accuracy - Digit: {history[-1]['digit']*100:.2f}%, Color: {history[-1]['color']*100:.2f}%, Parity: {history[-1]['parity']*100:.2f}%")
    return save_path

def load_checkpoint(checkpoint_path, model, logger, device):
    """
    Load checkpoint and restore model, ridge classifiers, history, and logger.
    """
    print(f"\n📂 LOADING CHECKPOINT: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    classifiers = checkpoint['classifiers']  # Ridge classifiers
    
    history = checkpoint['history']
    logger.data = checkpoint['logger_data']
    start_epoch = checkpoint['epoch'] + 1
    
    print(f"✅ CHECKPOINT LOADED!")
    print(f"   Resuming from epoch: {start_epoch}")
    print(f"   Previous accuracy - Digit: {history[-1]['digit']*100:.2f}%, Color: {history[-1]['color']*100:.2f}%, Parity: {history[-1]['parity']*100:.2f}%")
    
    return start_epoch, history, classifiers

# ============================================================================
# MAIN WITH CHECKPOINT RESUME CAPABILITY
# ============================================================================

def main(start_epoch=1, end_epoch=2, checkpoint_path=None):
    """
    Main training function with checkpoint support.
    
    Parameters:
    -----------
    start_epoch : int
        Starting epoch number (1-indexed)
    end_epoch : int
        Ending epoch number (inclusive)
    checkpoint_path : str or None
        Path to checkpoint file to resume from. If None, starts fresh.
    
  
    """
    
    print("\n" + "="*70)
    print(f"TRAINING CONFIGURATION: Epochs {start_epoch} to {end_epoch}")
    if checkpoint_path:
        print(f"RESUMING FROM: {checkpoint_path}")
    else:
        print("STARTING FRESH TRAINING")
    print("="*70)
    
    # Initialize logger
   logger = ComprehensiveLogger(save_dir='./output/')

    
    # Prepare RGB data (NO grayscale conversion!)
    print("\n🎨 Preparing RGB colored MNIST data...")
    X_train, y_train_d, y_train_c, y_train_p, X_test, y_test_d, y_test_c, y_test_p = prepare_colored_mnist_rgb(seed=42)
    
    batch_size = 32
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    np.random.seed(42)
    
    train_loader = DataLoader(
        TensorDataset(X_train, y_train_d, y_train_c, y_train_p),
        batch_size=batch_size, shuffle=True, drop_last=True
    )
    test_loader = DataLoader(
        TensorDataset(X_test, y_test_d, y_test_c, y_test_p),
        batch_size=batch_size, shuffle=False, drop_last=True
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n✓ Using device: {device}\n")
    
    print("="*70)
    print("INITIALIZING RGB MODEL")
    print("="*70)
    
    model = TemporalPoolingReservoir_RGB(batch_size, logger=logger).to(device)
    
    # NO ADAM OPTIMIZER - We use Ridge Regression (paper's approach!)
    # NO criterion needed - Ridge Regression handles classification directly
    
    # Initialize history and classifiers
    history = []
    classifiers = None  # Will be trained with Ridge Regression
    actual_start_epoch = start_epoch
    
    # Load checkpoint if resuming
    if checkpoint_path:
        try:
            actual_start_epoch, history, classifiers = load_checkpoint(checkpoint_path, model, logger, device)
            # Verify we're starting from the right epoch
            if actual_start_epoch != start_epoch:
                print(f"\n⚠️  WARNING: Checkpoint indicates epoch {actual_start_epoch}, but you specified {start_epoch}")
                print(f"   Using checkpoint epoch: {actual_start_epoch}")
                start_epoch = actual_start_epoch
            print(f"✓ Loaded Ridge classifiers from checkpoint")
        except Exception as e:
            print(f"\n❌ ERROR loading checkpoint: {e}")
            print("   Starting fresh instead...")
            history = []
            classifiers = None
    
    # Log training configuration
    logger.log_configuration({
        'optimizer': 'Ridge Regression (Paper Approach)',
        'regularization_alpha_digit': 1e-3,
        'regularization_alpha_color': 1e-3,
        'regularization_alpha_parity': 1e-4,
        'total_epochs_planned': end_epoch,
        'current_run_epochs': f'{start_epoch}-{end_epoch}',
        'batch_size': batch_size,
        'tasks': ['digit', 'color', 'parity'],
        'train_size': len(X_train),
        'test_size': len(X_test),
        'encoding_strategy': 'Strategy_1_Grayscale',
        'resuming_from_checkpoint': checkpoint_path is not None,
        'expected_accuracy': 'Digit 90-96% (paper baseline), Color 40-60%, Parity 95-98%'
    })
    
    print("\n" + "="*70)
    print(f"TRAINING WITH RIDGE REGRESSION (PAPER'S APPROACH)")
    print("="*70)
    print("✓ No epochs needed - Ridge Regression trains directly!")
    print("✓ Extracting reservoir features from all training data...")
    print("✓ Training 3 independent classifiers (digit, color, parity)...")
    print("="*70)
    
    # RIDGE REGRESSION TRAINING (replaces all epoch-based Adam training!)
    training_start = time.time()
    
    # Train with Ridge Regression - returns trained classifiers and results
    clf_digit, clf_color, clf_parity, results = train_with_ridge_regression(
        model, train_loader, test_loader, device, logger, epoch_num=end_epoch
    )
    
    classifiers = {
        'digit': clf_digit,
        'color': clf_color,
        'parity': clf_parity
    }
    
    # Add to history
    history.append(results)
    
    training_duration = time.time() - training_start
    
    # Log results
    logger.log_epoch(end_epoch, results, training_duration)
    
    # Print results with RGB comparison
    print(f"\nRESULTS:")
    print(f"  Digit:  {results['digit']*100:.2f}%")
    
    # Color classification feedback
    color_status = "🎉 EXCELLENT!" if results['color'] > 0.50 else "📈 Good!" if results['color'] > 0.40 else "⚠️  Below target"
    print(f"  Color:  {results['color']*100:.2f}% (random baseline: 10%) {color_status}")
    
    print(f"  Parity: {results['parity']*100:.2f}%")
    print(f"  Training Duration: {training_duration:.2f}s")
    
    if results['digit'] >= 0.90:
        print("🎯 EXCELLENT DIGIT ACCURACY (matches paper!)")
    if results['color'] >= 0.50:
        print("🌈 GOOD COLOR ACCURACY!")
    if results['parity'] >= 0.95:
        print("✅ OUTSTANDING PARITY ACCURACY!")
    
    # Save checkpoint
    checkpoint_path = f'./output/checkpoint_ridge_regression.pth'
    save_checkpoint(end_epoch, model, classifiers, history, logger, checkpoint_path)
    logger.log_model_checkpoint(end_epoch, checkpoint_path, results)
    
    # Final results summary
    print("\n" + "="*70)
    print(f"RIDGE REGRESSION TRAINING COMPLETE")
    print("="*70)
    final = history[-1]
    print(f"Final Results:")
    print(f"  Digit:  {final['digit']*100:.2f}% (Paper baseline: 96%)")
    print(f"  Color:  {final['color']*100:.2f}% (Random baseline: 10%)")
    print(f"  Parity: {final['parity']*100:.2f}% (Random baseline: 50%)")
    
    print(f"\n✅ Grayscale encoding (Strategy 1) advantages:")
    print(f"   - Matches paper's architecture exactly (784 neurons, 15,680 features)")
    print(f"   - Single pass through reservoir (fast!)")
    print(f"   - Paper's validated approach (96% digit accuracy)")
    print(f"   - Ridge Regression readout (scientifically rigorous)")
    
    print(f"\n🎉 TRAINING COMPLETE!")
    print(f"   Ridge Regression trains in one pass - no need for multiple runs!")
    print(f"   Reservoir is fixed (not trained), only readout classifiers trained.")
    
    print("="*70)
    
    # Save final model
    final_model_path = f'./output/model_ridge_regression.pth'
    torch.save(model.state_dict(), final_model_path)
    print(f"\n✓ Model saved: {final_model_path}")
    
    # Save Ridge classifiers
    classifiers_path = f'./output/ridge_classifiers.pkl'
    with open(classifiers_path, 'wb') as f:
        pickle.dump(classifiers, f)
    print(f"✓ Ridge classifiers saved: {classifiers_path}")
    
    # Create training plot
    plot_results(history, method='Ridge Regression (RGB)', save_path=f'./output/ridge_regression_results.png')

    
    # Save logger data
    logger.save_all()
    
    print(f"\n✅ RIDGE REGRESSION TRAINING COMPLETE!")
    
    return history, logger, classifiers

# ============================================================================
# CONVENIENCE FUNCTION FOR RIDGE REGRESSION TRAINING
# ============================================================================

def run_ridge_regression_training():
    """
    Run complete training with Ridge Regression (Paper's approach).
    
    Uses Strategy 1 (Grayscale-Equivalent encoding):
    - RGB → Grayscale conversion (0.299*R + 0.587*G + 0.114*B)
    - Single pass through 784-neuron reservoir
    - 15,680 features (matches paper!)
    - Ridge Regression for multi-task readout
    
    No multiple sessions needed! Ridge Regression trains in one pass.
    This is the scientifically validated approach from Zhang et al.
    
    Returns:
    --------
    history : list
        Training history with results
    logger : ComprehensiveLogger
        Logger with all training data
    classifiers : dict
        Trained Ridge classifiers for digit, color, parity
    """
    print("\n" + "🚀"*35)
    print("RIDGE REGRESSION TRAINING (PAPER'S APPROACH)")
    print("🚀"*35)
    print("\n✓ Scientifically validated method")
    print("✓ No epochs needed - trains in one pass!")
    print("✓ Linear readout (biologically plausible)")
    print("✓ Standard in reservoir computing literature")
    print("\n" + "🚀"*35 + "\n")
    
    return main(start_epoch=1, end_epoch=1, checkpoint_path=None)

if __name__ == "__main__":
    # ==========================================================================
    # USAGE INSTRUCTIONS FOR RIDGE REGRESSION
    # ==========================================================================
    # 
    # SIMPLE - JUST ONE COMMAND:
    # 
    #     history, logger, classifiers = run_ridge_regression_training()
    # 
    # That's it! Ridge Regression trains everything in one pass.
    # No need for multiple sessions like with Adam optimizer.
    # 
    # Expected results (Strategy 1 - Grayscale encoding):
    #   - Digit:  90-96% (matches paper baseline!)
    #   - Color:  40-60% (grayscale encoding limits color info)
    #   - Parity: 95-98% (easiest task)
    # 
    # Note: Color accuracy lower than RGB preservation approach,
    # but matches paper's validated architecture exactly.
    # ==========================================================================
    
    # Default: Run Ridge Regression Training
    print("\n" + "="*70)
    print("STARTING RIDGE REGRESSION TRAINING (GRAYSCALE ENCODING)")
    print("="*70)
    
    history, logger, classifiers = run_ridge_regression_training()

