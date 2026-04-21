from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import resnet18


class CIFARLeNet(nn.Module):
    def __init__(self, use_batch_norm: bool = False, dropout_p: float = 0.0) -> None:
        super().__init__()
        conv1_layers: list[nn.Module] = [nn.Conv2d(3, 6, kernel_size=5)]
        if use_batch_norm:
            conv1_layers.append(nn.BatchNorm2d(6))
        conv1_layers.extend([nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2)])

        conv2_layers: list[nn.Module] = [nn.Conv2d(6, 16, kernel_size=5)]
        if use_batch_norm:
            conv2_layers.append(nn.BatchNorm2d(16))
        conv2_layers.extend([nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2)])

        self.features = nn.Sequential(
            *conv1_layers,
            *conv2_layers,
        )
        classifier_layers: list[nn.Module] = [
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(inplace=True),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
        ]
        if dropout_p > 0.0:
            classifier_layers.append(nn.Dropout(p=dropout_p))
        classifier_layers.append(nn.Linear(84, 10))
        self.classifier = nn.Sequential(*classifier_layers)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return torch.flatten(x, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.extract_features(x))


class ResNet18CIFAR(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = resnet18(weights=None)
        self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.backbone.maxpool = nn.Identity()
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 10)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        x = self.backbone.avgpool(x)
        return torch.flatten(x, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone.fc(self.extract_features(x))


def build_model(model_key: str) -> nn.Module:
    if model_key == "lenet_baseline":
        return CIFARLeNet(use_batch_norm=False, dropout_p=0.0)
    if model_key == "lenet_bn_dropout":
        return CIFARLeNet(use_batch_norm=True, dropout_p=0.3)
    if model_key == "resnet18_cifar":
        return ResNet18CIFAR()
    raise ValueError(f"Unsupported model key: {model_key}")


def count_trainable_parameters(model: nn.Module) -> int:
    return int(sum(param.numel() for param in model.parameters() if param.requires_grad))
