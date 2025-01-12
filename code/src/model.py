import torch
from torch import nn
from torch.optim import Optimizer, AdamW
import lightning as L

from torchvision.utils import _log_api_usage_once
from torchvision.models.resnet import BasicBlock, Bottleneck, conv1x1

from typing import Callable, Type, Literal
from torch import Tensor


def count_trainable_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class ResNet(nn.Module):
    """ResNet model.

    https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
    """
    def __init__(
        self,
        block: Type[BasicBlock | Bottleneck] = BasicBlock,
        layers: list[int] = [2, 2, 2],
        inplanes: int = 4,  # Changed to accept inplanes as argument.
        num_classes: int = 9,
        output_patch_size: int = 1,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: list[bool] | None = None,
        norm_layer: Callable[..., nn.Module] | None = None,
    ) -> None:
        super().__init__()
        _log_api_usage_once(self)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = inplanes
        self.num_classes = num_classes
        if output_patch_size % 2 != 1:
            raise ValueError(
                f'`outout_block_size` must be an odd number, is {output_patch_size}.'
            )
        self.output_patch_size = output_patch_size
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        # Changed the first layer to accept 4 channels.
        self.conv1 = nn.Conv2d(4, self.inplanes, kernel_size=3, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, inplanes, layers[0])
        self.layer2 = self._make_layer(block, inplanes * 2, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, inplanes * 4, layers[2], stride=1, dilate=replace_stride_with_dilation[1])
        self.avgpool = nn.AdaptiveAvgPool2d((output_patch_size, output_patch_size))
        self.fc = nn.Conv2d(self.inplanes * block.expansion, num_classes, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
        self,
        block: Type[BasicBlock | Bottleneck],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        if x.shape[-1] < self.output_patch_size:
            raise AssertionError(
                f'the output spatial dimension of the tensor after ResNet layers ({x.shape[-2:]}) is smaller than the '
                f'`output_patch_size` ({self.output_patch_size}). Increase the input spatial dimension or reduce '
                f'`output_patch_size` to {x.shape[-1]}.'
            )

        x = self.avgpool(x)

        x = self.fc(x)

        # Reshape from (channels, classes, output_patch_size, output_patch_size)
        # to (channels, output_patch_size, output_patch_size, classes)
        x = x.permute(0, 2, 3, 1)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


class LightningResNet(L.LightningModule):
    def __init__(
            self,
            num_classes: int,
            output_patch_size: int,
            learning_rate: float = 0.001,
            weight_decay: float = 0,
            use_class_weights: bool = False,
            class_weights: Tensor | None = None):

        super().__init__()

        self.save_hyperparameters()

        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.class_weights = class_weights

        if use_class_weights:
            self.class_weights = self.class_weights
        elif not use_class_weights:
            self.class_weights = None
        else:
            raise ValueError(
                f'`use_class_weights` must be a boolean, is {use_class_weights}.'
            )

        self.model = ResNet(inplanes=4, num_classes=num_classes, output_patch_size=output_patch_size)

        self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)

    def calculate_accuracy(self, y_hat: Tensor, y: Tensor) -> float:
        y_hat_classes = y_hat.argmax(dim=-1)
        correct = (y_hat_classes == y).sum()
        total = y.numel()
        return correct / total

    def common_step(self, batch, mode: Literal['train', 'valid', 'test']):

        # (channels, output_patch_size, output_patch_size, classes)
        y_hat = self.model(batch['x'])
        y = batch['y']

        y_hat_flat = y_hat.flatten(0, 2)
        y_flat = y.flatten(0, 2)

        # Merge channels, output_patch_size, output_patch_size into single dimension.
        loss = self.criterion(y_hat_flat, y_flat)

        accuracy = self.calculate_accuracy(y_hat_flat, y_flat)

        self.log_dict({
            f'{mode}_loss': loss,
            f'{mode}_acc': accuracy
        }, logger=True, on_step=(mode == 'train'), on_epoch=True)

        return y_hat, loss

    def training_step(self, batch, batch_idx) -> Tensor:
        _, loss, = self.common_step(batch, mode='train')

        return loss

    def validation_step(self, batch, batch_idx) -> Tensor:
        _, loss, = self.common_step(batch, mode='valid')

        return loss

    def test_step(self, batch, batch_idx) -> Tensor:
        _, loss, = self.common_step(batch, mode='test')

        return loss

    def predict_step(self, batch, batch_idx) -> tuple[Tensor, dict[str, int]]:

        # Softmax across 2nd dim (the classes).
        y_hat = self.model(batch['x']).softmax(-1)

        del batch['x']

        return y_hat, batch

    def configure_optimizers(self) -> Optimizer:

        optimizer = AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        return optimizer
