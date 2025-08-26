import numpy as np
import pytorch_lightning as pl
import scanpy as sc
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from torchcfm.optimal_transport import OTPlanSampler
from pytorch_lightning.utilities.combined_loader import CombinedLoader


class TemporalDataModule(pl.LightningDataModule):
    def __init__(
        self,
        args,
        skipped_datapoint=-1,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.args = args  # Store args for access in _prepare_data
        self.data_type = args.data_type
        self.data_path = args.data_path
        self.batch_size = args.batch_size
        self.split_ratios = args.split_ratios
        self.max_dim = args.dim
        self.whiten = args.whiten
        self.skipped_datapoint = skipped_datapoint
        self._prepare_data()

    def _prepare_data(self):
        self.train_dataloaders = []
        self.val_dataloaders = []
        self.test_dataloaders = []
        self.metric_samples_dataloaders = []

        if self.data_type == "scrna":
            ds, labels, unique_labels = custom_load_dataset(
                self.data_path,
                max_dim=self.max_dim,
            )
        elif self.data_type == "arch":
            ds, labels, unique_labels = generate_arch_data()
        elif self.data_type == "gaussian":
            num_points = getattr(self.args, 'num_points', 5000)
            time_points = getattr(self.args, 'time_points', None)
            # Convert integer time_points to linspace
            if isinstance(time_points, int):
                import numpy as np
                time_points = np.linspace(0, 1, time_points).tolist()
            ds, labels, unique_labels = generate_gaussian_data(num_points=num_points, time_points=time_points)
        elif self.data_type == "knot":
            num_points = getattr(self.args, 'num_points', 5000)
            time_points = getattr(self.args, 'time_points', None)
            # Convert integer time_points to linspace
            if isinstance(time_points, int):
                import numpy as np
                time_points = np.linspace(0, 1, time_points).tolist()
            ds, labels, unique_labels = generate_knot_data(num_points=num_points, time_points=time_points)
        elif self.data_type == "sphere":
            ds, labels, unique_labels = generate_sphere_data()
        else:
            raise ValueError("Data type not recognized")
        if self.whiten:
            self.scaler = StandardScaler()
            ds = self.scaler.fit_transform(ds)

        ds_tensor = torch.tensor(ds, dtype=torch.float32)
        label_to_numeric = {label: idx for idx, label in enumerate(unique_labels)}
        frame_indices = {
            label_to_numeric[label]: (labels == label).nonzero()[0]
            for label in unique_labels
        }
        self.num_timesteps = len(unique_labels)

        min_frame_size = min([len(indices) for indices in frame_indices.values()])
        for label, indices in frame_indices.items():
            frame_data = ds_tensor[indices]
            split_index = int(len(frame_data) * self.split_ratios[0])

            if len(frame_data) - split_index < self.batch_size:
                split_index = len(frame_data) - self.batch_size
            shuffled_indices = torch.randperm(len(frame_data))
            frame_data = frame_data[shuffled_indices]
            train_data = frame_data[:split_index]
            val_data = frame_data[split_index:]
            self.train_dataloaders.append(
                DataLoader(
                    train_data,
                    batch_size=self.batch_size,
                    shuffle=True,
                    drop_last=True,
                )
            )
            self.val_dataloaders.append(
                DataLoader(
                    val_data,
                    batch_size=self.batch_size,
                    shuffle=False,
                    drop_last=True,
                )
            )
            self.test_dataloaders.append(
                DataLoader(
                    frame_data,
                    batch_size=frame_data.shape[0],
                    shuffle=False,
                    drop_last=False,
                )
            )
            self.metric_samples_dataloaders.append(
                DataLoader(
                    frame_data,
                    batch_size=min_frame_size,
                    shuffle=True,
                    drop_last=False,
                )
            )

    def train_dataloader(self):
        combined_loaders = {
            "train_samples": CombinedLoader(self.train_dataloaders, mode="min_size"),
            "metric_samples": CombinedLoader(
                self.metric_samples_dataloaders, mode="min_size"
            ),
        }
        return CombinedLoader(combined_loaders, mode="max_size_cycle")

    def val_dataloader(self):
        combined_loaders = {
            "val_samples": CombinedLoader(self.val_dataloaders, mode="min_size"),
            "metric_samples": CombinedLoader(
                self.metric_samples_dataloaders, mode="min_size"
            ),
        }

        return CombinedLoader(combined_loaders, mode="max_size_cycle")

    def test_dataloader(self):
        return CombinedLoader(self.test_dataloaders, "max_size")


def adata_dataset(
    path: str,
    embed_name: str = "X_pca",
    label_name: str = "day",
    max_dim: int = 100,
):
    """Load Single Cell dataset from h5ad file using scanpy."""

    adata = sc.read_h5ad(path)
    labels = adata.obs[label_name].astype("category")
    ulabels = labels.cat.categories
    data = adata.obsm[embed_name][:, :max_dim]

    return (data, labels.to_numpy(), ulabels.to_numpy())


def tnet_dataset(
    path: str,
    embed_name: str = "pcs",
    label_name: str = "sample_labels",
    max_dim: int = 100,
):
    """Load Single Cell dataset from npz file."""

    data_dict = np.load(path, allow_pickle=True)
    data = data_dict[embed_name][:, :max_dim]
    labels = data_dict[label_name]
    unique_labels = np.unique(labels)
    return data, labels, unique_labels


def custom_load_dataset(path: str, max_dim: int = 100):
    if path.endswith("h5ad"):
        return adata_dataset(path, max_dim=max_dim)
    if path.endswith("npz"):
        return tnet_dataset(path, max_dim=max_dim)
    raise NotImplementedError(f"File extension not supported for path: {path}")


def generate_arch_data(num_points: int = 5000):
    """Generate synthetic data for the arch dataset."""

    time_0_samples = np.abs(
        np.random.normal(loc=0, scale=1 / (2 * np.pi), size=num_points)
    )
    time_2_samples = 1 - np.abs(
        np.random.normal(loc=0, scale=1 / (2 * np.pi), size=num_points)
    )

    x0_ot, x1_ot = OTPlanSampler(method="exact").sample_plan(
        torch.tensor(time_0_samples).unsqueeze(0),
        torch.tensor(time_2_samples).unsqueeze(0),
        replace=False,
    )
    x0_ot, x1_ot = x0_ot.numpy().flatten(), x1_ot.numpy().flatten()
    time_1_samples = (x0_ot + x1_ot) / 2

    # Mapping to a semi-circle
    angles_0 = np.pi * (1 - time_0_samples)
    angles_1 = np.pi * (1 - time_1_samples)
    angles_2 = np.pi * (1 - time_2_samples)

    x_0 = np.cos(angles_0)
    y_0 = np.sin(angles_0)
    x_1 = np.cos(angles_1)
    y_1 = np.sin(angles_1)
    x_2 = np.cos(angles_2)
    y_2 = np.sin(angles_2)

    # Adding Gaussian noise
    radius_noise_0 = np.random.normal(0, 0.1, size=num_points)
    radius_noise_1 = np.random.normal(0, 0.1, size=num_points)
    radius_noise_2 = np.random.normal(0, 0.1, size=num_points)

    x_0 = (1 + radius_noise_0) * x_0
    y_0 = (1 + radius_noise_0) * y_0
    x_1 = (1 + radius_noise_1) * x_1
    y_1 = (1 + radius_noise_1) * y_1
    x_2 = (1 + radius_noise_2) * x_2
    y_2 = (1 + radius_noise_2) * y_2

    # Combining points and creating labels
    points_0 = np.column_stack((x_0, y_0))
    points_1 = np.column_stack((x_1, y_1))
    points_2 = np.column_stack((x_2, y_2))

    points = np.concatenate([points_0, points_1, points_2])
    labels = np.array([0] * num_points + [1] * num_points + [2] * num_points)

    # Returning the dataset, labels, and unique labels
    unique_labels = np.unique(labels)
    return points, labels, unique_labels


def x_fun_partial(t_param, std):
    """Generate x coordinates for knot trajectory.
    
    Args:
        t_param: Parameter values in range [0, 3] 
        std: noise standard deviation
    """
    x_vals = []
    
    for t in t_param:
        if t <= 1:
            # First segment: linear
            x = 3 * (t - 0.5)
        elif t <= 2:
            # Second segment: cosine loop
            x = np.cos(2 * np.pi * (t - 1.75))
        else:
            # Third segment: linear
            x = 3 * (t - 2.5)
        x_vals.append(x)
    
    x_vals = np.array(x_vals)
    return x_vals + np.random.randn(*x_vals.shape) * std


def y_fun_partial(t_param, std):
    """Generate y coordinates for knot trajectory.
    
    Args:
        t_param: Parameter values in range [0, 3]
        std: noise standard deviation
    """
    y_vals = []
    
    for t in t_param:
        if t <= 1:
            # First segment: tanh curve
            y = -np.tanh(5 * (t - 0.5)) / 2 + 0.5
        elif t <= 2:
            # Second segment: sine loop
            y = np.sin(2 * np.pi * (t - 1.75)) + 1
        else:
            # Third segment: tanh curve
            y = np.tanh(5 * (t - 2.5)) / 2 + 0.5
        y_vals.append(y)
    
    y_vals = np.array(y_vals)
    return y_vals + np.random.randn(*y_vals.shape) * std


def x_fun(t, std):
    t = 3 * (t / t.max()) - 1.5
    size = t.shape[0]
    assert size % 3 == 0

    t1, t2, t3 = t[:size // 3], t[size // 3:-size // 3], t[-size // 3:]

    x1 = 3 * (t1 + 0.5)
    x2 = np.cos(2 * np.pi * (t2 - 0.75))
    x3 = 3 * (t3 - 0.5)

    x = np.concatenate([x1, x2, x3])
    return x + np.random.randn(*x.shape) * std 


def y_fun(t, std):
    t = 3 * (t / t.max()) - 1.5
    size = t.shape[0]
    assert size % 3 == 0

    t1, t2, t3 = t[:size // 3], t[size // 3:-size // 3], t[-size // 3:]
    y1 = - np.tanh(5 * (t1 + 1)) / 2 + 0.5
    y2 = np.sin(2 * np.pi * (t2 - 0.75)) + 1
    y3 = np.tanh(5 * (t3 - 1)) / 2 + 0.5

    y = np.concatenate([y1, y2, y3])
    return y + np.random.randn(*y.shape) * std

def loop_distribution(size, std):
    assert size % 3 == 0
    t = np.linspace(0, 3, size)
    xt = np.stack([x_fun(t, std), y_fun(t, std)]).T
    x0 = np.random.randn(2, size).T * std + np.array([-3, 1])
    x1 = np.random.randn(2, size).T * std + np.array([3, 1])
    return x0, xt, x1, t / 3


def generate_knot_data(num_points:int=5000, time_points: list = None):
    """Generate synthetic knot data at specified timesteps.
    
    Args:
        num_points: Number of points per timestep
        time_points: List of time values (default: [0, 0.5, 1.0])
    """
    if time_points is None:
        time_points = [0.0, 0.5, 1.0]
    
    # For the full trajectory, we need more points to sample from
    trajectory_points = max(10000, num_points * 3)
    if trajectory_points % 3 != 0:
        trajectory_points = ((trajectory_points // 3) + 1) * 3
    
    # Generate the full knot trajectory with many points
    X0_full, Xt_full, X1_full, t_params = loop_distribution(trajectory_points, std=0.0)
    
    all_points = []
    all_labels = []
    
    for i, t in enumerate(time_points):
        if t == 0.0:
            # Start distribution: sample from X0
            indices = np.random.choice(len(X0_full), size=num_points, replace=False)
            points = X0_full[indices] + np.random.randn(num_points, 2) * 0.1
        elif t == 1.0 and len(time_points) > 2:
            # For t=1.0, we want the final segment of the trajectory, not just the end distribution
            # This connects the last intermediate time to the end
            prev_t = time_points[-2]  # Get the time point before 1.0
            
            # Sample from the final segment [prev_t, 1.0]
            mask = (t_params > prev_t) & (t_params <= t)
            valid_indices = np.where(mask)[0]
            
            if len(valid_indices) >= num_points:
                indices = np.random.choice(valid_indices, size=num_points, replace=False)
            else:
                indices = np.random.choice(valid_indices, size=num_points, replace=True)
            
            points = Xt_full[indices] + np.random.randn(num_points, 2) * 0.1
        elif t == 1.0:
            # Fallback for when we only have start and end (no intermediate points)
            indices = np.random.choice(len(X1_full), size=num_points, replace=False)
            points = X1_full[indices] + np.random.randn(num_points, 2) * 0.1
        else:
            # Intermediate time: sample from a specific segment of the trajectory
            # We want non-overlapping segments for each time point
            
            # Find the previous time point
            prev_t = 0.0
            for j in range(i):
                if time_points[j] < t:
                    prev_t = time_points[j]
            
            # Sample from the segment [prev_t, t]
            mask = (t_params > prev_t) & (t_params <= t)
            valid_indices = np.where(mask)[0]
            
            if len(valid_indices) == 0:
                # Fallback: if no points in range, sample around t
                center_idx = int(t * len(t_params))
                valid_indices = np.arange(max(0, center_idx - 100), min(len(t_params), center_idx + 100))
            
            if len(valid_indices) >= num_points:
                # Sample without replacement if we have enough points
                indices = np.random.choice(valid_indices, size=num_points, replace=False)
            else:
                # Sample with replacement if we don't have enough
                indices = np.random.choice(valid_indices, size=num_points, replace=True)
            
            points = Xt_full[indices] + np.random.randn(num_points, 2) * 0.1
        
        all_points.append(points)
        all_labels.extend([i] * num_points)
    
    points = np.concatenate(all_points)
    labels = np.array(all_labels)
    unique_labels = np.unique(labels)
    
    return points, labels, unique_labels


def generate_gaussian_data(num_points: int = 5000, time_points: list = None):
    """Generate 1D Gaussian synthetic data at specified timesteps.
    
    Args:
        num_points: Number of points per timestep
        time_points: List of time values (default: [0, 0.5, 1.0])
    """
    if time_points is None:
        time_points = [0.0, 0.5, 1.0]
    
    n_times = len(time_points)
    all_points = []
    all_labels = []
    
    for i, t in enumerate(time_points):
        if t == 0.0:
            # Time 0: Single Gaussian centered at 0
            x = np.random.normal(0., 0.3, num_points)
        elif t == 1.0:
            # Time 1: Single Gaussian centered at 0
            x = np.random.normal(0., 0.3, num_points)
        else:
            # Intermediate times: progressively split into multiple Gaussians
            if t <= 0.5:
                # From t=0 to t=0.5: progressively split from 1 to 3 Gaussians
                split_factor = t / 0.5  # 0 to 1 as t goes from 0 to 0.5
                n_per_component = num_points // 3
                remaining = num_points - 3 * n_per_component
                
                # Three Gaussian components that spread apart
                spread = 1.5 * split_factor
                component_1 = np.random.normal(-spread, 0.2 + 0.1*(1-split_factor), n_per_component)
                component_2 = np.random.normal(0.0, 0.2 + 0.1*(1-split_factor), n_per_component)
                component_3 = np.random.normal(spread, 0.2 + 0.1*(1-split_factor), n_per_component + remaining)
                
                x = np.concatenate([component_1, component_2, component_3])
            else:
                # From t=0.5 to t=1: progressively merge from 3 to 1 Gaussian
                merge_factor = (t - 0.5) / 0.5  # 0 to 1 as t goes from 0.5 to 1
                n_per_component = num_points // 3
                remaining = num_points - 3 * n_per_component
                
                # Three components that merge together
                spread = 1.5 * (1 - merge_factor)
                component_1 = np.random.normal(-spread, 0.2 + 0.1*merge_factor, n_per_component)
                component_2 = np.random.normal(0.0, 0.2 + 0.1*merge_factor, n_per_component)
                component_3 = np.random.normal(spread, 0.2 + 0.1*merge_factor, n_per_component + remaining)
                
                x = np.concatenate([component_1, component_2, component_3])
            np.random.shuffle(x)
        
        points = x.reshape(-1, 1)
        all_points.append(points)
        all_labels.extend([i] * num_points)
    
    points = np.concatenate(all_points)
    labels = np.array(all_labels)
    unique_labels = np.unique(labels)
    
    return points, labels, unique_labels


def generate_sphere_data(num_points: int = 5000):
    time_0_samples = np.abs(
        np.random.normal(loc=0, scale=1 / (2 * np.pi), size=num_points)
    )
    time_2_samples = 1 - np.abs(
        np.random.normal(loc=0, scale=1 / (2 * np.pi), size=num_points)
    )

    x0_ot, x1_ot = OTPlanSampler(method="exact").sample_plan(
        torch.tensor(time_0_samples).unsqueeze(0),
        torch.tensor(time_2_samples).unsqueeze(0),
        replace=False,
    )
    x0_ot, x1_ot = x0_ot.numpy().flatten(), x1_ot.numpy().flatten()
    time_1_samples = (x0_ot + x1_ot) / 2

    phi_0 = np.pi * time_0_samples
    phi_1 = np.pi * time_1_samples
    phi_2 = np.pi * time_2_samples

    theta_0 = 2 * np.pi * np.random.rand(num_points)
    theta_1 = 2 * np.pi * np.random.rand(num_points)
    theta_2 = 2 * np.pi * np.random.rand(num_points)

    x_0 = np.sin(phi_0) * np.cos(theta_0)
    y_0 = np.sin(phi_0) * np.sin(theta_0)
    z_0 = np.cos(phi_0)
    x_1 = np.sin(phi_1) * np.cos(theta_1)
    y_1 = np.sin(phi_1) * np.sin(theta_1)
    z_1 = np.cos(phi_1)
    x_2 = np.sin(phi_2) * np.cos(theta_2)
    y_2 = np.sin(phi_2) * np.sin(theta_2)
    z_2 = np.cos(phi_2)

    # Combining points and creating labels
    points_0 = np.column_stack((x_0, y_0, z_0))
    points_1 = np.column_stack((x_1, y_1, z_1))
    points_2 = np.column_stack((x_2, y_2, z_2))

    points = np.concatenate([points_0, points_1, points_2])
    labels = np.array([0] * num_points + [1] * num_points + [2] * num_points)

    unique_labels = np.unique(labels)
    return points, labels, unique_labels
