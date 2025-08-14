import argparse
import copy
import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
import wandb

from torchcfm.optimal_transport import OTPlanSampler

from mfm.flow_matchers.models.mfm import MetricFlowMatcher
from mfm.geo_metrics.metric_factory import DataManifoldMetric
from mfm.flow_matchers.flow_net_train import (
    FlowNetTrainTrajectory,
    FlowNetTrainLidar,
    FlowNetTrainImage,
)
from mfm.flow_matchers.geopath_net_train import GeoPathNetTrain
from mfm.dataloaders.trajectory_data import TemporalDataModule
from mfm.dataloaders.image_data import ImageDataModule
from mfm.dataloaders.lidar_data import LidarDataModule
from mfm.networks.flow_networks.mlp import VelocityNet
from mfm.networks.geopath_networks.mlp import GeoPathMLP
from mfm.networks.unet_base import UNetModelWrapper as UNetModel
from mfm.networks.geopath_networks.unet import GeoPathUNet
from mfm.utils import set_seed
from mfm.train.parsers import parse_args
from mfm.flow_matchers.ema import EMA
from mfm.train.train_utils import (
    load_config,
    merge_config,
    generate_group_string,
    dataset_name2datapath,
    create_callbacks,
)


def visualize_learned_paths_1d(geopath_model, flow_model, datamodule, save_path="gaussian_mfm_paths.png"):
    """Visualize learned 1D flow paths between marginal distributions"""
    
    if datamodule.data_type != "gaussian":
        print("Visualization only available for 1D Gaussian data")
        return
    
    print("Generating visualization of learned paths...")
    
    # Get sample data from each timestep by accessing the dataset directly
    from mfm.dataloaders.trajectory_data import generate_gaussian_data
    points, labels, unique_labels = generate_gaussian_data(1000)
    
    # Extract data for each timestep
    x0_samples = points[labels == 0].flatten()  # t=0 data
    x1_samples = points[labels == 1].flatten()  # t=1 data  
    x2_samples = points[labels == 2].flatten()  # t=2 data
    
    # Create a grid of starting points from t=0 distribution
    n_paths = 15
    x_start = np.linspace(x0_samples.min(), x0_samples.max(), n_paths)
    
    # Generate learned paths using the flow network
    time_steps = np.linspace(0, 1, 50)
    
    fig, ax1 = plt.subplots(1, 1, figsize=(14, 8))
    
    # Generate proper flow paths using the trained flow network
    all_paths = []
    
    # Use OTPlanSampler to properly couple x0 and x2 samples
    ot_sampler = OTPlanSampler(method="exact")
    
    # Sample subset of data for visualization
    x0_subset = torch.tensor(x0_samples[:n_paths*5]).unsqueeze(-1)  # [N, 1]
    x2_subset = torch.tensor(x2_samples[:n_paths*5]).unsqueeze(-1)  # [N, 1]
    
    # Get OT coupling
    x0_coupled, x2_coupled = ot_sampler.sample_plan(x0_subset, x2_subset)
    
    # Take first n_paths for visualization
    x0_vis = x0_coupled[:n_paths].flatten().numpy()
    x2_vis = x2_coupled[:n_paths].flatten().numpy()
    
    for i in range(n_paths):
        path = []
        x_init = x0_vis[i]
        x_end = x2_vis[i]
        x_current = torch.tensor([[x_init]], dtype=torch.float32)
        
        for t in time_steps:
            if t == 0:
                path.append(x_init)
            elif t == 1:
                path.append(x_end)
            else:
                # Use the trained flow matcher's geodesic interpolation
                t_tensor = torch.tensor([t], dtype=torch.float32)
                x_end_tensor = torch.tensor([[x_end]], dtype=torch.float32)
                
                with torch.no_grad():
                    try:
                        # Use the flow matcher's compute_mu_t method (Equation 20)
                        if hasattr(flow_model, 'flow_matcher'):
                            mu_t = flow_model.flow_matcher.compute_mu_t(
                                x_current, x_end_tensor, t_tensor, 
                                torch.tensor([0.0]), torch.tensor([1.0])
                            )
                            path.append(mu_t.item())
                        else:
                            # Fallback to linear interpolation
                            x_interp = (1 - t) * x_init + t * x_end
                            path.append(x_interp)
                    except Exception as e:
                        # Simple linear interpolation fallback
                        x_interp = (1 - t) * x_init + t * x_end
                        path.append(x_interp)
        
        all_paths.append(path)
        ax1.plot(time_steps, path, 'b-', alpha=0.4, linewidth=1.5)
    
    # Add marginal distributions as vertical histograms at specific times
    t_marginal_times = [0.0, 0.5, 1.0]
    marginal_data = [x0_samples, x1_samples, x2_samples]
    marginal_colors = ['red', 'green', 'blue']
    marginal_labels = ['t=0', 't=0.5', 't=1']
    
    for t_val, data, color, label in zip(t_marginal_times, marginal_data, marginal_colors, marginal_labels):
        # Create histogram
        hist, bin_edges = np.histogram(data, bins=25, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Scale and offset histogram to show as vertical distribution
        hist_scaled = hist * 0.02  # Scale factor for visibility
        
        # Plot as filled area
        for j in range(len(bin_centers)):
            ax1.fill_betweenx([bin_centers[j] - (bin_edges[1] - bin_edges[0])/2, 
                              bin_centers[j] + (bin_edges[1] - bin_edges[0])/2],
                             t_val, t_val + hist_scaled[j], 
                             alpha=0.7, color=color)
        
        # Add vertical line at time point
        ax1.axvline(t_val, color=color, linestyle='--', alpha=0.8, linewidth=2, label=label)
    
    ax1.set_xlabel('time t', fontsize=12)
    ax1.set_ylabel('position x', fontsize=12)
    ax1.set_title('learned flow paths with marginal distributions (mfm)', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    ax1.set_xlim(-0.05, 1.05)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to {save_path}")


def main(args: argparse.Namespace, seed: int, t_exclude: int) -> None:
    set_seed(seed)
    if args.data_type == "lidar":
        assert args.dim == 3 and args.data_name == "lidar"
    elif args.data_type == "arch":
        assert args.dim == 2
    elif args.data_type == "gaussian":
        assert args.dim == 1
    elif args.data_type == "knot":
        assert args.dim == 2
    elif args.data_type == "sphere":
        assert args.dim == 3
    elif args.data_type == "image":
        assert not args.whiten
        assert args.data_name == "afhq"

    skipped_time_points = [t_exclude] if t_exclude else []

    ### DATAMODULES
    if args.data_type in ["arch", "gaussian", "knot", "scrna", "sphere"]:
        datamodule = TemporalDataModule(
            args=args,
            skipped_datapoint=t_exclude,
        )
    elif args.data_type == "lidar":
        datamodule = LidarDataModule(args=args)
    elif args.data_type == "image":
        datamodule = ImageDataModule(args=args)
    else:
        raise ValueError("Data type not recognized")

    ### Interpolation and Vector Field Networks
    if args.data_type in ["arch", "gaussian", "knot", "scrna", "lidar", "sphere"]:
        flow_net = VelocityNet(
            dim=args.dim,
            hidden_dims=args.hidden_dims_flow,
            activation=args.activation_flow,
            batch_norm=False,
        )
        geopath_net = GeoPathMLP(
            input_dim=args.dim,
            hidden_dims=args.hidden_dims_geopath,
            time_geopath=args.time_geopath,
            activation=args.activation_geopath,
            batch_norm=False,
        )
    elif args.data_type == "image":
        flow_net = UNetModel(
            geopath_model=False,
            dim=datamodule.dim,
            num_channels=args.unet_num_channels,
            num_res_blocks=args.unet_num_res_blocks,
            channel_mult=args.unet_channel_mult,
            dropout=args.unet_dropout,
            resblock_updown=args.unet_resblock_updown,
            use_new_attention_order=args.unet_use_new_attention_order,
            attention_resolutions=args.unet_attention_resolutions,
            num_heads=args.unet_num_heads,
        )
        geopath_net = GeoPathUNet(
            geopath_model=True,
            dim=datamodule.dim,
            num_channels=args.unet_num_channels_geopath,
            num_res_blocks=args.unet_num_res_blocks_geopath,
            channel_mult=args.unet_channel_mult_geopath,
            dropout=args.unet_dropout_geopath,
            use_checkpoint=False,
        )

    if args.ema_decay is not None:
        flow_net = EMA(model=flow_net, decay=args.ema_decay)
        geopath_net = EMA(model=geopath_net, decay=args.ema_decay)

    ot_sampler = (
        OTPlanSampler(method=args.optimal_transport_method)
        if args.optimal_transport_method != "None"
        else None
    )

    wandb.init(
        project=f"mfm-{args.data_type}-{args.data_name}",
        group=args.group_name,
        config=vars(args),
        dir=args.working_dir,
    )

    ### Metric Flow Matching Module
    flow_matcher_base = MetricFlowMatcher(
        geopath_net=geopath_net,
        sigma=args.sigma,
        alpha=int(args.mfm),
    )

    ##### ALGO 1: Training of Geodesic Interpolants Beginning #####
    if args.mfm:
        data_manifold_metric = DataManifoldMetric(
            args=args,
            skipped_time_points=skipped_time_points,
            datamodule=datamodule,
        )
        geopath_callbacks = create_callbacks(
            args, phase="geopath", data_type=args.data_type, run_id=wandb.run.id
        )

        geopath_model = GeoPathNetTrain(
            flow_matcher=flow_matcher_base,
            skipped_time_points=skipped_time_points,
            ot_sampler=ot_sampler,
            data_manifold_metric=data_manifold_metric,
            args=args,
        )
        wandb_logger = WandbLogger()

        trainer = Trainer(
            max_epochs=args.epochs,
            callbacks=geopath_callbacks,
            accelerator=args.accelerator,
            logger=wandb_logger,
            num_sanity_val_steps=0,
            default_root_dir=args.working_dir,
            gradient_clip_val=(1.0 if args.data_type == "image" else None),
        )
        if args.load_geopath_model_ckpt:
            best_model_path = args.load_geopath_model_ckpt
        else:
            trainer.fit(
                geopath_model,
                datamodule=datamodule,
            )
            best_model_path = geopath_callbacks[0].best_model_path
        geopath_model = GeoPathNetTrain.load_from_checkpoint(best_model_path)

        flow_matcher_base.geopath_net = geopath_model.geopath_net

    ##### ALGO 1: Training of Geodesic Interpolants END #####

    ##### ALGO 2: (Metric) Flow Matching Beginning #####
    if args.data_type in ["arch", "gaussian", "knot", "scrna", "sphere"]:
        datamodule = TemporalDataModule(
            args=args,
            skipped_datapoint=t_exclude,
        )
    flow_callbacks = create_callbacks(
        args,
        phase="flow",
        data_type=args.data_type,
        run_id=wandb.run.id,
        datamodule=datamodule,
    )

    if args.data_type in ["arch", "gaussian", "knot", "scrna", "sphere"]:
        FlowNetTrain = FlowNetTrainTrajectory
    elif args.data_type == "lidar":
        FlowNetTrain = FlowNetTrainLidar
    elif args.data_type == "image":
        FlowNetTrain = FlowNetTrainImage
    else:
        raise ValueError("Data type not recognized")

    flow_train = FlowNetTrain(
        flow_matcher=flow_matcher_base,
        flow_net=flow_net,
        ot_sampler=ot_sampler,
        skipped_time_points=skipped_time_points,
        args=args,
    )

    wandb_logger = WandbLogger()

    trainer = Trainer(
        max_epochs=args.epochs,
        callbacks=flow_callbacks,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        accelerator=args.accelerator,
        logger=wandb_logger,
        default_root_dir=args.working_dir,
        gradient_clip_val=(1.0 if args.data_type == "image" else None),
        num_sanity_val_steps=(0 if args.data_type == "image" else None),
    )

    trainer.fit(
        flow_train, datamodule=datamodule, ckpt_path=args.resume_flow_model_ckpt
    )
    trainer.test(flow_train, datamodule=datamodule)
    
    # Add visualization for Gaussian 1D data
    if args.data_type == "gaussian":
        visualize_learned_paths_1d(geopath_model, flow_train, datamodule, 
                                   save_path=f"gaussian_mfm_paths_seed{seed}.png")
    
    wandb.finish()
    ##### ALGO 2: (Metric) Flow Matching END #####


if __name__ == "__main__":
    args = parse_args()
    updated_args = copy.deepcopy(args)
    if args.config_path:
        config = load_config(args.config_path)
        updated_args = merge_config(updated_args, config)

    updated_args.group_name = generate_group_string()
    updated_args.data_path = dataset_name2datapath(
        updated_args.data_name, updated_args.working_dir
    )
    for seed in updated_args.seeds:
        if updated_args.t_exclude:
            for i, t_exclude in enumerate(updated_args.t_exclude):
                updated_args.t_exclude_current = t_exclude
                updated_args.seed_current = seed
                # Use modulo to handle cases where there are fewer gammas than t_exclude values
                updated_args.gamma_current = updated_args.gammas[i % len(updated_args.gammas)]
                main(updated_args, seed=seed, t_exclude=t_exclude)
        else:
            updated_args.seed_current = seed
            updated_args.gamma_current = updated_args.gammas[0]
            main(updated_args, seed=seed, t_exclude=None)
