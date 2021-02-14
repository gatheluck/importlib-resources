import importlib.resources
import pathlib
from typing import Callable, List, Literal, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


def _weights_init(m: nn.Module) -> None:
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal(m.weight)


class LambdaLayer(nn.Module):
    def __init__(self, lambd: Callable[[torch.Tensor], torch.Tensor]) -> None:
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self, in_planes: int, planes: int, stride: int = 1, option: str = "A"
    ) -> None:
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        if stride != 1 or in_planes != planes:
            if option == "A":
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(  # type: ignore
                    lambda x: F.pad(
                        x[:, :, ::2, ::2],
                        (0, 0, 0, 0, planes // 4, planes // 4),
                        "constant",
                        0,
                    )
                )
            elif option == "B":
                self.shortcut = nn.Sequential(  # type: ignore
                    nn.Conv2d(
                        in_planes,
                        self.expansion * planes,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    nn.BatchNorm2d(self.expansion * planes),
                )
            else:
                raise ValueError(f"{option} is not supported.")
        else:
            self.shortcut = nn.Sequential()  # type: ignore

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(
        self, block: Type[nn.Module], num_blocks: List[int], num_classes: int = 10
    ) -> None:
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(
        self, block: Type[nn.Module], planes: int, num_blocks: int, stride: int
    ) -> nn.Module:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))  # type: ignore
            self.in_planes = planes * block.expansion  # type: ignore

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        rep = out.view(out.size(0), -1)
        logit = self.linear(rep)

        return logit


MODE = Literal["raw", "dunder_file", "importlib"]


def resnet56(
    pretrained: bool = False, num_classes: int = 10, mode: MODE = "importlib"
) -> nn.Module:

    arch = ResNet(BasicBlock, [9, 9, 9], num_classes=num_classes)

    if pretrained:
        if mode == "raw":
            weight_path = pathlib.Path("archs/weight/pretrained.pth")
            print(f"loading weight from {weight_path}.")
            arch.load_state_dict(torch.load(weight_path))
        elif mode == "dunder_file":
            weight_path = (
                pathlib.Path(__file__).parent.parent / "weight" / "pretrained.pth"
            )
            print(f"loading weight from {weight_path}.")
            arch.load_state_dict(torch.load(weight_path))
        elif mode == "importlib":
            with importlib.resources.path(
                "archs.weight", "pretrained.pth"
            ) as weight_path:
                print(f"loading weight from {weight_path}.")
                arch.load_state_dict(torch.load(weight_path))
        else:
            raise ValueError(f"{mode} is not supported.")

    return arch
