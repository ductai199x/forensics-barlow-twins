from typing import *

import torch
from pytorch_lightning.core.lightning import LightningModule
from torch import Tensor, nn, optim
from torch.nn import functional as F
from torch.nn.common_types import _size_2_t
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _pair
from torchmetrics import AUROC, Accuracy


class ConstrainedConv2d(_ConvNd):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",  # TODO: refine this type
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        kernel_size_ = _pair(kernel_size)
        stride_ = _pair(stride)
        padding_ = padding if isinstance(padding, str) else _pair(padding)
        dilation_ = _pair(dilation)
        super(ConstrainedConv2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size_,
            stride_,
            padding_,
            dilation_,
            False,
            _pair(0),
            groups,
            bias,
            padding_mode,
            **factory_kwargs,
        )

    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        if self.padding_mode != "zeros":
            return F.conv2d(
                F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                weight,
                bias,
                self.stride,
                _pair(0),
                self.dilation,
                self.groups,
            )
        return F.conv2d(input, weight, bias, self.stride, self.padding, self.dilation, self.groups)

    def constrain_weight(self) -> None:
        shape = self.weight.shape  # [0]: filter, [1]: channel, [2]: width, [3]: height
        tmp_w = self.weight.data
        center_pos = (shape[-2] * shape[-1]) // 2
        for filtr in range(shape[0]):
            for channel in range(shape[1]):
                krnl = tmp_w[filtr, channel, :, :]
                krnl = krnl.view(-1)

                krnl[center_pos] = 0
                krnl = krnl * 10000
                krnl = krnl / torch.sum(krnl)
                krnl[center_pos] = -1
                krnl = torch.reshape(krnl, (shape[-2], shape[-1]))
                tmp_w[filtr, channel, :, :] = krnl

        self.weight = nn.Parameter(tmp_w)

    def forward(self, input: Tensor) -> Tensor:
        self.constrain_weight()
        return self._conv_forward(input, self.weight, self.bias)


class MISLnet(nn.Module):
    def __init__(self, classification_head=True, num_output_feature=200, num_classes=70):
        super().__init__()
        self.model = nn.Sequential(
            *(
                [
                    nn.Conv2d(3, 3, kernel_size=5, stride=1, padding="valid"),
                    nn.Conv2d(3, 96, kernel_size=7, stride=2, padding=(5, 5)),
                    nn.BatchNorm2d(96),
                    nn.Tanh(),
                    nn.MaxPool2d(3, stride=2, padding=(1, 1)),
                    nn.Conv2d(96, 64, kernel_size=5, stride=1, padding="same"),
                    nn.BatchNorm2d(64),
                    nn.Tanh(),
                    nn.MaxPool2d(3, stride=2, padding=(1, 1)),
                    nn.Conv2d(64, 64, kernel_size=5, stride=1, padding="same"),
                    nn.BatchNorm2d(64),
                    nn.Tanh(),
                    nn.MaxPool2d(3, stride=2),
                    nn.Conv2d(64, 128, kernel_size=1, stride=1),
                    nn.BatchNorm2d(128),
                    nn.Tanh(),
                    nn.AvgPool2d(3, stride=2, padding=(1, 1)),
                    nn.Flatten(),
                    nn.LazyLinear(num_output_feature),
                ]
                + (
                    [
                        nn.Tanh(),
                        nn.LazyLinear(num_output_feature),
                        nn.Tanh(),
                        nn.Linear(num_output_feature, num_classes),
                    ]
                    if classification_head
                    else []
                )
            )
        )
        self.init_weights()

    def init_weights(self):
        for layer in self.children():
            if isinstance(layer, nn.Conv2d):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0.0)

    def forward(self, x):
        return self.model(x)


class Dict2Class:
    def __init__(self, my_dict):
        for key in my_dict:
            setattr(self, key, my_dict[key])


class MISLnetPLWrapper(LightningModule):
    default_args = {
        "input_size": (128, 128),
        "output_dim": 1024,
        "num_classes": 70,
        "lr": 1e-3,
        "momentum": 0.95,
        "decay_rate": 0.5,
        "decay_step": 4,
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

        self.model = MISLnet(True, self.args.output_dim, self.args.num_classes)

        self.example_input_array = torch.randn(1, 3, *self.args.input_size)

        self.train_acc = Accuracy()
        self.val_acc = Accuracy()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)

        self.log("train_loss", loss)
        self.log(
            "train_acc_step",
            self.train_acc(logits, y),
            prog_bar=True,
            on_step=True,
            on_epoch=False,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)

        self.log("val_loss", loss)
        self.log("val_acc_step", self.val_acc(logits, y))

    def on_train_epoch_start(self) -> None:
        self.train_acc.reset()

    def on_validation_epoch_start(self) -> None:
        self.val_acc.reset()

    def on_train_epoch_end(self) -> None:
        self.log("train_acc_epoch", self.train_acc.compute(), prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        self.log("val_acc_epoch", self.val_acc.compute(), prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        # steplr = torch.optim.lr_scheduler.StepLR(
        #     optimizer, step_size=self.args.decay_step, gamma=self.args.decay_rate
        # )
        lr_sched = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, "min", self.args.decay_rate, 2, min_lr=1e-6, verbose=True
            ),
            "monitor": "val_loss",
            "interval": "epoch",
            "frequency": 1,
        }
        return [optimizer], [lr_sched]
