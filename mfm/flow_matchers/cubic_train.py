import torch
import pytorch_lightning as pl
import numpy as np
from scipy import interpolate

from torchcfm.optimal_transport import OTPlanSampler


def _minibatch_couple_marginals(observed_x: torch.Tensor) -> torch.Tensor:
    """
    Align a minibatch of paths across consecutive timepoints using OT maps.

    observed_x: [batch_size, num_times, dim]
    returns aligned tensor with the same shape
    """
    bs, J, _ = observed_x.shape
    otplan = OTPlanSampler('exact')

    aligned = torch.zeros_like(observed_x)
    idxs = torch.arange(bs, device=observed_x.device)
    aligned[:, 0] = observed_x[:, 0]

    for j in range(J - 1):
        pi_np = otplan.get_map(observed_x[:, j], observed_x[:, j + 1])  # (bs, bs) numpy
        # Convert plan to torch.float32 on the same device (MPS doesn't support float64)
        pi_t = torch.from_numpy(pi_np).to(device=observed_x.device, dtype=torch.float32)  # (bs, bs)
        probs = pi_t[idxs]
        probs = probs / torch.sum(probs, dim=1, keepdim=True)
        idxs = torch.multinomial(probs, num_samples=1).squeeze(1)
        aligned[:, j + 1] = observed_x[idxs, j + 1]

    return aligned


class CubicInterpolantTrain(pl.LightningModule):
    """
    Pretraining stage: supervise the flow_net on cubic-spline interpolants
    constructed from observed timepoints (e.g., t in {0, 0.5, 1}).

    Loss: MSE between predicted velocity and cubic spline derivative dI/dt.
    """

    def __init__(self, flow_net, args, observed_time: float = 0.5):
        super().__init__()
        self.save_hyperparameters(ignore=['flow_net'])
        self.flow_net = flow_net
        self.args = args
        self.observed_time = observed_time

        # Optim settings (reuse flow_* args)
        self.optimizer_name = getattr(args, 'flow_optimizer', 'adamw')
        self.lr = getattr(args, 'flow_lr', 1e-3)
        self.weight_decay = getattr(args, 'flow_weight_decay', 1e-5)

    def _prepare_batch_triplets(self, batch):
        # batch contains either "train_samples", "val_samples", or "test_samples"
        if "train_samples" in batch:
            main_batch = batch["train_samples"][0]
        elif "val_samples" in batch:
            main_batch = batch["val_samples"][0]
        elif "test_samples" in batch:
            main_batch = batch["test_samples"][0]
        else:
            raise KeyError("Expected one of 'train_samples', 'val_samples', or 'test_samples' in batch")
        x0 = main_batch[0]
        x1 = main_batch[-1]
        # pick middle time as observed (assumes odd number of times)
        mid_idx = len(main_batch) // 2
        xt_obs = main_batch[mid_idx]

        # Squeeze possible extra dims (keep feature dimension)
        def _fix(x):
            return torch.squeeze(x, dim=0) if x.dim() > 2 else x
        x0 = _fix(x0).to(self.device)
        xt_obs = _fix(xt_obs).to(self.device)
        x1 = _fix(x1).to(self.device)

        # Ensure equal batch sizes via trimming to min
        bs = min(x0.shape[0], xt_obs.shape[0], x1.shape[0])
        x0, xt_obs, x1 = x0[:bs], xt_obs[:bs], x1[:bs]

        observed_x = torch.stack([x0, xt_obs, x1], dim=1)  # [bs, 3, d]
        aligned = _minibatch_couple_marginals(observed_x)
        return aligned  # [bs, 3, d]

    def _sample_targets(self, aligned: torch.Tensor):
        """
        Build cubic spline per path and sample (t, xt, dxt) targets.
        aligned: [bs, 3, d]
        Returns t: [bs, 1], xt: [bs, d], dxt: [bs, d]
        """
        bs, _, d = aligned.shape
        device = aligned.device

        # times [0, observed, 1]
        t_obs = np.array([0.0, self.observed_time, 1.0], dtype=np.float64)

        # random t in (0,1)
        t = torch.rand(bs, 1, device=device)

        xt = torch.empty(bs, d, device=device)
        dxt = torch.empty(bs, d, device=device)

        aligned_np = aligned.detach().cpu().numpy()
        t_np = t.detach().cpu().numpy().reshape(-1)

        for b in range(bs):
            cs = interpolate.CubicSpline(t_obs, aligned_np[b])
            xt[b] = torch.from_numpy(cs(t_np[b])).to(device, dtype=torch.float32)
            dxt[b] = torch.from_numpy(cs(t_np[b], nu=1)).to(device, dtype=torch.float32)

        return t, xt, dxt

    def training_step(self, batch, batch_idx):
        aligned = self._prepare_batch_triplets(batch)
        t, xt, dxt = self._sample_targets(aligned)

        pred = self.flow_net(t.squeeze(-1), xt)
        loss = torch.nn.functional.mse_loss(pred, dxt)
        self.log("Cubic/train_mse", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        aligned = self._prepare_batch_triplets(batch)
        t, xt, dxt = self._sample_targets(aligned)
        with torch.no_grad():
            pred = self.flow_net(t.squeeze(-1), xt)
            loss = torch.nn.functional.mse_loss(pred, dxt)
        self.log("Cubic/val_mse", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        if self.optimizer_name == "adam":
            optimizer = torch.optim.Adam(self.flow_net.parameters(), lr=self.lr)
        else:
            optimizer = torch.optim.AdamW(self.flow_net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer


