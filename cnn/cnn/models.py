import torchvision

def load_model(model: str, num_classes: int = 1000, pretrained: bool = False):
    match model:
        case "resnet18":
            return torchvision.models.resnet18(
                num_classes=num_classes,
                weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None,
            )
        case "resnet50":
            return torchvision.models.resnet50(
                num_classes=num_classes,
                weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None,
            )
        case "vgg16_bn":
            return torchvision.models.vgg16_bn(
                num_classes=num_classes,
                weights=torchvision.models.VGG16_BN_Weights.IMAGENET1K_V1 if pretrained else None,
            )
        case "vgg16":
            return torchvision.models.vgg16(
                num_classes=num_classes,
                weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None,
            )
        case "vit_base_16":
            return torchvision.models.vit_b_16(
                num_classes=num_classes,
                weights=torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None,
            )
        case _:
            raise ValueError(f"Unknown model: {model}")
