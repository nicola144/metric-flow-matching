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
            ds, labels, unique_labels = generate_gaussian_data(num_points=num_points)
        elif self.data_type == "knot":
            num_points = getattr(self.args, 'num_points', 5000)
            ds, labels, unique_labels = generate_knot_data(num_points=num_points)
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


def generate_knot_data(num_points:int=5000):
    """Generate synthetic knot data in the format expected by MFM."""
    
    # Ensure num_points is divisible by 3
    if num_points % 3 != 0:
        num_points = ((num_points // 3) + 1) * 3
    
    X0, Xt, X1, times = loop_distribution(num_points, std=0.1)
    
    # Create points and labels in the same format as arch_data
    points_0 = X0  # t=0 data
    points_1 = Xt  # t=0.5 data (knot trajectory)
    points_2 = X1  # t=1 data
    
    # Combine all points
    points = np.concatenate([points_0, points_1, points_2])
    labels = np.array([0] * num_points + [1] * num_points + [2] * num_points)
    
    unique_labels = np.unique(labels)
    return points, labels, unique_labels


def generate_gaussian_data(num_points: int = 5000):
    """Generate 1D Gaussian synthetic data at 3 timesteps with mixture at t=0.5."""
    
    # Time 0: Single Gaussian centered at -2
    x_0 = np.random.normal(0., 0.3, num_points)
    
    # Time 2: Single Gaussian centered at 2  
    x_2 = np.random.normal(0., 0.3, num_points)
    
    # Time 1 (t=0.5)
    n_per_component = num_points // 3
    remaining = num_points - 3 * n_per_component
    
    # Three Gaussian components at different locations
    component_1 = np.random.normal(-1.5, 0.2, n_per_component)  # Left component
    component_2 = np.random.normal(0.0, 0.2, n_per_component)   # Center component  
    component_3 = np.random.normal(1.5, 0.2, n_per_component + remaining)  # Right component
    
    # Combine the three components
    x_1 = np.concatenate([component_1, component_2, component_3])
    np.random.shuffle(x_1)
    
    # Combine points as 1D data (reshape to column vector)
    points_0 = x_0.reshape(-1, 1)
    points_1 = x_1.reshape(-1, 1)
    points_2 = x_2.reshape(-1, 1)
    
    points = np.concatenate([points_0, points_1, points_2])
    labels = np.array([0] * num_points + [1] * num_points + [2] * num_points)
    
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
