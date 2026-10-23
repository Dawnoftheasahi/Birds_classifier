"""Model definitions for the web app.

BirdCNN is copied from the training notebook so the Space does not depend on it.
ResNet-50 and ConvNeXt-Tiny are rebuilt with timm (pretrained=False: we load our own
fine-tuned weights instead of the ImageNet ones).
"""
import timm
import torch
import torch.nn as nn


class ConvBNAct(nn.Sequential):
    """Conv(k x k) -> BatchNorm -> ReLU."""
    def __init__(self, c_in, c_out, k=3, stride=1):
        super().__init__(nn.Conv2d(c_in, c_out, k, stride, k // 2, bias=False),
                         nn.BatchNorm2d(c_out), nn.ReLU(inplace=True))


class ResidualBlock(nn.Module):
    """Basic residual block: out = ReLU( BN(conv(ConvBNAct(x))) + shortcut(x) )."""
    def __init__(self, c_in, c_out, stride=1):
        super().__init__()
        self.conv1 = ConvBNAct(c_in, c_out, 3, stride)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        self.shortcut = (nn.Identity() if stride == 1 and c_in == c_out else
                         nn.Sequential(nn.Conv2d(c_in, c_out, 1, stride, bias=False), nn.BatchNorm2d(c_out)))
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn2(self.conv2(self.conv1(x))) + self.shortcut(x))


class BirdCNN(nn.Module):
    """Compact residual CNN trained from scratch (same forward API as timm models)."""
    def __init__(self, num_classes=200, widths=(64, 128, 256, 512), depths=(1, 2, 2, 1), drop_rate=0.0):
        super().__init__()
        self.stem = nn.Sequential(ConvBNAct(3, 32, 3, 2), ConvBNAct(32, widths[0], 3, 1),
                                  nn.MaxPool2d(3, 2, 1))
        stages, c_in = [], widths[0]
        for i, (w, d) in enumerate(zip(widths, depths)):
            blocks = [ResidualBlock(c_in, w, stride=1 if i == 0 else 2)]
            blocks += [ResidualBlock(w, w) for _ in range(d - 1)]
            stages.append(nn.Sequential(*blocks))
            c_in = w
        self.stages = nn.Sequential(*stages)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(drop_rate)
        self.fc = nn.Linear(c_in, num_classes)
        self.num_features = c_in

    def forward_features(self, x):
        return self.stages(self.stem(x))

    def forward_head(self, x, pre_logits=False):
        x = self.pool(x).flatten(1)
        return x if pre_logits else self.fc(self.drop(x))

    def forward(self, x):
        return self.forward_head(self.forward_features(x))

    def get_classifier(self):
        return self.fc


def load_model(arch, weights_path, num_classes):
    """Rebuild an architecture and load the exported (float16) weights as float32, in eval mode."""
    if arch == 'birdcnn':
        model = BirdCNN(num_classes)
    else:
        model = timm.create_model(arch, pretrained=False, num_classes=num_classes)
    state = torch.load(weights_path, map_location='cpu', weights_only=True)
    model.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in state.items()})
    for p in model.parameters():
        p.requires_grad_(False)        # inference only; Grad-CAM needs gradients w.r.t. features, not weights
    return model.eval()
