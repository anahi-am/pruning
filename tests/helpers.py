import torch
from torch import nn


class TinyConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.f = -1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(self.conv(x))


class TinyDWConv(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            channels, channels, 3, padding=1, groups=channels, bias=False
        )
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(self.conv(x))


class TinyC3k2(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.cv1 = TinyConv(in_channels, out_channels)
        self.cv2 = TinyConv(out_channels * 2, out_channels)
        self.f = -1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv1(x)
        return self.cv2(torch.cat((y, y), dim=1))


class Concat(nn.Module):
    def __init__(self, sources: list[int]) -> None:
        super().__init__()
        self.f = list(sources)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class Detect(nn.Module):
    def __init__(self, sources: list[int], branch_channels: int = 4) -> None:
        super().__init__()
        self.f = list(sources)
        self.cv2 = nn.ModuleList()
        self.cv3 = nn.ModuleList()
        self.one2one_cv2 = nn.ModuleList()
        self.one2one_cv3 = nn.ModuleList()
        for _ in sources:
            self.cv2.append(
                nn.Sequential(TinyConv(branch_channels, 2), nn.Conv2d(2, 1, 1))
            )
            self.one2one_cv2.append(
                nn.Sequential(TinyConv(branch_channels, 2), nn.Conv2d(2, 1, 1))
            )
            cv3_branch = nn.Sequential(
                nn.Sequential(
                    TinyDWConv(branch_channels), TinyConv(branch_channels, 2)
                ),
                nn.Conv2d(2, 2, 1),
            )
            self.cv3.append(cv3_branch)
            self.one2one_cv3.append(
                nn.Sequential(
                    nn.Sequential(
                        TinyDWConv(branch_channels), TinyConv(branch_channels, 2)
                    ),
                    nn.Conv2d(2, 2, 1),
                )
            )


class TinyModel(nn.Module):
    def __init__(self, layers: list[nn.Module]) -> None:
        super().__init__()
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs: list[torch.Tensor] = []
        for idx, layer in enumerate(self.model):
            f = getattr(layer, "f", -1)
            if isinstance(f, list):
                sources = [idx - 1 if source == -1 else source for source in f]
                inputs = [outputs[source] for source in sources]
                x = torch.cat(inputs, dim=1)
            else:
                x = layer(outputs[-1] if outputs else x)
            outputs.append(x)
        return x


class FakeYOLO:
    def __init__(self, detection_model: TinyModel) -> None:
        self.model = detection_model


def make_mask(num_channels: int, num_to_keep: int) -> torch.BoolTensor:
    mask = torch.zeros(num_channels, dtype=torch.bool)
    mask[:num_to_keep] = True
    return mask
