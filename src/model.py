import torch
import torch.nn as nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.transform import GeneralizedRCNNTransform


def create_radar_detector(num_classes: int = 2, pretrained_backbone: bool = True) -> nn.Module:
    """
    Cria uma Faster R-CNN com backbone ResNet-50-FPN customizada para 1 canal.
    """
    # 1. Carrega o modelo Faster R-CNN
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained_backbone else None
    model = fasterrcnn_resnet50_fpn(weights=weights)

    # 2. Modifica a primeira convolução do backbone (ResNet conv1) de 3 para 1 canal
    old_conv = model.backbone.body.conv1
    new_conv = nn.Conv2d(
        in_channels=1,                 # 1 canal (radar)
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=False
    )

    if pretrained_backbone:
        with torch.no_grad():
            new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))

    model.backbone.body.conv1 = new_conv

    # 3. CORREÇÃO DO ERRO: Substitui a camada interna de transform do torchvision
    # Define média e desvio padrão para 1 canal único e fixa o tamanho no shape nativo (256)
    # Como nossos dados já chegam normalizados [0.0, 1.0], usamos mean=0.0 e std=1.0
    model.transform = GeneralizedRCNNTransform(
        min_size=256,
        max_size=256,
        image_mean=[0.0],
        image_std=[1.0]
    )

    # 4. Substitui a cabeça de predição (FastRCNNPredictor)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


if __name__ == "__main__":
    # Teste rápido de forward pass
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo de teste: {device}")

    model = create_radar_detector(num_classes=2, pretrained_backbone=True).to(device)
    model.eval()

    # Cria um lote simulado de 2 frames de radar: shape (B, C, H, W) = (2, 1, 256, 256)
    dummy_input = torch.rand(2, 1, 256, 256, dtype=torch.float32).to(device)

    with torch.no_grad():
        predictions = model(dummy_input)

    print("\n[OK] Modelo inicializado com sucesso!")
    print(f"Previsões retornadas: {len(predictions)} frames.")
    print("Chaves por predição:", predictions[0].keys())