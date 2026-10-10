import os
import torch
from torch.utils.data import DataLoader, random_split
from radar_dataset import RadarDataset, collate_fn_detection
from src.model import create_radar_detector


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    total_loss = 0.0

    for batch_idx, (images, targets) in enumerate(data_loader):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # No modo train, a Faster R-CNN retorna um dicionário com as perdas
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        # Gradient clipping para estabilizar o treino
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        total_loss += losses.item()

    avg_loss = total_loss / len(data_loader)
    print(f"Epoch [{epoch+1}] | Loss Treino: {avg_loss:.4f}")
    return avg_loss


@torch.no_grad()
def evaluate_loss(model, data_loader, device):
    """
    Calcula a perda no conjunto de validação mantendo model.train() apenas
    internamente para forçar a extração do loss_dict pelo torchvision.
    """
    model.train()
    val_loss = 0.0

    for images, targets in data_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        val_loss += losses.item()

    return val_loss / len(data_loader) if len(data_loader) > 0 else 0.0


def main():
    # 1. Configurações
    ARRAYS_DIR = "./data/processed/arrays"
    LABELS_DIR = "./data/processed/labels"
    CHECKPOINT_DIR = "./checkpoints"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    BATCH_SIZE = 4
    EPOCHS = 15
    LR = 1e-4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Treinando em: {DEVICE}")

    # 2. Carrega Dataset e divide Treino/Validação
    full_dataset = RadarDataset(arrays_dir=ARRAYS_DIR, labels_dir=LABELS_DIR)
    val_size = max(1, int(0.2 * len(full_dataset)))
    train_size = len(full_dataset) - val_size

    train_data, val_data = random_split(
        full_dataset, 
        [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)
    )

    # num_workers=0 para evitar problemas de multiprocessing
    train_loader = DataLoader(
        train_data, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, collate_fn=collate_fn_detection
    )
    val_loader = DataLoader(
        val_data, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, collate_fn=collate_fn_detection
    )

    print(f"Frames de treino: {len(train_data)} | Frames de validação: {len(val_data)}")

    # 3. Inicializa modelo e otimizador
    model = create_radar_detector(num_classes=2, pretrained_backbone=True).to(DEVICE)
    
    # Otimizador AdamW com decaimento de peso
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_loss = float("inf")

    # 4. Loop de treinamento
    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(model, optimizer, train_loader, DEVICE, epoch)
        val_loss = evaluate_loss(model, val_loader, DEVICE)
        lr_scheduler.step()

        print(f"          | Loss Validação: {val_loss:.4f}")

        # Salva o melhor checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(CHECKPOINT_DIR, "radar_detector_best.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"          -> Checkpoint salvo em: {best_model_path}")

    print("\n[Treinamento Finalizado com Sucesso]")


if __name__ == "__main__":
    main()