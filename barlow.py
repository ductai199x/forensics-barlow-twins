from typing import *

import torch
import torch.nn as nn
import torch.nn.functional as F

from mislnet import MISLnet
from pytorch_lightning import LightningModule


def off_diagonal(x: torch.Tensor) -> torch.Tensor:
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def get_output_shape(model: nn.Module, example_shape: torch.Tensor) -> torch.Tensor:
    return torch.tensor(model(torch.rand(*example_shape)).shape).to(torch.int32)


class BarlowTwins(nn.Module):
    def __init__(self, fe, input_size=(128, 128), fe_output_dim=1024, proj_output_dim=2048):
        super().__init__()

        self.fe = nn.Sequential(*(list(fe.model.children())[0][:-2]))

        fe_out_dim = get_output_shape(self.fe, [1, 3, *input_size])
        assert fe_out_dim.equal(torch.tensor([1, fe_output_dim]).to(torch.int32))

        self.fe_output_dim = fe_output_dim
        self.proj_output_dim = proj_output_dim

        self.projector = nn.Sequential(
            nn.Linear(self.fe_output_dim, self.proj_output_dim, bias=False),
            nn.BatchNorm1d(self.proj_output_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.proj_output_dim, self.proj_output_dim, bias=False),
            nn.BatchNorm1d(self.proj_output_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.proj_output_dim, self.proj_output_dim, bias=False),
        )

        for layer in self.projector.children():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)

    def forward(self, x):
        with torch.no_grad():
            z = self.fe(x)
        z = self.projector(z)

        return z


class Dict2Class:
    def __init__(self, my_dict):
        for key in my_dict:
            setattr(self, key, my_dict[key])

class BarlowTwinsPLWrapper(LightningModule):
    default_args = {
        "fe": None,
        "input_size": (128, 128), "fe_output_dim": 1024, "proj_output_dim": 2048,
        "lr": 1e-3, "momentum": 0.90, "decay_rate": 5e-4, "alpha": 5e-3,
    }
    def __init__(self, args):
        super().__init__()

        if isinstance(args, dict):
            args_keys = set(args.keys())
            # fill in default values if some keys are missing
            keys_diff = set(self.default_args.keys()).difference(args_keys)
            for k in keys_diff:
                args_keys[k] = self.default_args[k]
            self.args = Dict2Class(args)
        else:
            raise ValueError(f"args must be a python-dict")

        assert isinstance(self.args.fe, nn.Module), "fe - feature extractor - argument must be a nn.Module"

        self.model = BarlowTwins(self.args.fe, self.args.input_size, self.args.fe_output_dim, self.args.proj_output_dim)

        self.example_input_array = torch.randn([1, 3, *self.args.input_size])

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch

        if x.shape[0] < 3:
            x = torch.cat(4 * [x])

        comb = torch.combinations(torch.arange(x.shape[0]), r=2)

        x1 = x[comb[:, 0]]
        x2 = x[comb[:, 1]]

        z1 = self(x1)
        z2 = self(x2)

        z1_norm = (z1 - z1.mean(0)) / z1.std(0) # NxD
        z2_norm = (z2 - z2.mean(0)) / z2.std(0) # NxD

        N, D = z1_norm.shape

        cc = (z1_norm.T @ z2_norm) / (N)

        # on_diag = torch.diagonal(cc).add_(-1).pow_(2).sum()
        # off_diag = off_diagonal(out.clone()).pow_(2).sum()
        # loss = on_diag + self.args.alpha * off_diag

        c_diff = (cc - torch.eye(D, device=self.device)).pow_(2) # DxD
        # multiply off-diagonal elems of c_diff by lambda
        c_diff[~torch.eye(D, dtype=bool)] *= self.args.alpha
        loss = c_diff.sum()

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        if x.shape[0] < 3:
            x = torch.cat(4 * [x])

        comb = torch.combinations(torch.arange(x.shape[0]), r=2)

        x1 = x[comb[:, 0]]
        x2 = x[comb[:, 1]]

        z1 = self(x1)
        z2 = self(x2)

        z1_norm = (z1 - z1.mean(0)) / z1.std(0) # NxD
        z2_norm = (z2 - z2.mean(0)) / z2.std(0) # NxD

        N, D = z1_norm.shape

        cc = (z1_norm.T @ z2_norm) / (N)

        # on_diag = torch.diagonal(cc).add_(-1).pow_(2).sum()
        # off_diag = off_diagonal(out.clone()).pow_(2).sum()
        # loss = on_diag + self.args.alpha * off_diag

        c_diff = (cc - torch.eye(D, device=self.device)).pow_(2).div_(D) # DxD
        # multiply off-diagonal elems of c_diff by lambda
        c_diff[~torch.eye(D, dtype=bool)] *= self.args.alpha
        loss = c_diff.sum()

        self.log("val_loss", loss)

    # learning rate warm-up
    def optimizer_steps(self,
                        epoch=None,
                        batch_idx=None,
                        optimizer=None,
                        optimizer_idx=None,
                        optimizer_closure=None,
                        on_tpu=None,
                        using_native_amp=None,
                        using_lbfgs=None):        
        steps_per_epc = 50000
        num_epc = 10
        max_steps = steps_per_epc * num_epc
        if self.trainer.global_step < max_steps:
            lr_scale = min(1., (self.trainer.global_step + 1) / max_steps)
            for pg in optimizer.param_groups:
                pg['lr'] = lr_scale * 1e-3

        # update params
        optimizer.step()
        optimizer.zero_grad()

    def configure_optimizers(self):
        optim = torch.optim.SGD(self.parameters(), lr=self.args.lr,
                                momentum=self.args.momentum, weight_decay=self.args.decay_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, 100)
        return [optim], [scheduler]

    # def configure_optimizers(self):
    #     optimizer = torch.optim.SGD(self.parameters(), lr=self.args.lr, momentum=0.97)
    #     steplr = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.args.decay_step, gamma=self.args.decay_rate)
    #     return [optimizer], [steplr]


if __name__ == "__main__":
    brltw = BarlowTwins()
    brltw = brltw.eval()

    example_data = torch.randn((1, 3, 256, 256))
    example_out = brltw(example_data).T @ brltw(example_data)
    print(example_out.shape)
