import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from typing import List, Tuple, Dict, Any, Optional

class RadarDataset(Dataset):
    """
    Dataset customizado em PyTorch para dados de radar meteorológico 2D.
    
    Lê matrizes normalizadas (.npy) de refletividade e seus respectivos
    rótulos YOLO (.txt), convertendo as caixas para o formato absoluto
    esperado por modelos PyTorch: [xmin, ymin, xmax, ymax].
    """
    def __init__(
        self,
        arrays_dir: str,
        labels_dir: str,
        transform: Optional[Any] = None
    ):
        self.arrays_dir = arrays_dir
        self.labels_dir = labels_dir
        self.transform = transform

        # Lista e ordena todos os arquivos de array
        self.array_files = sorted(glob.glob(os.path.join(arrays_dir, "*.npy")))
        if len(self.array_files) == 0:
            raise FileNotFoundError(f"Nenhum arquivo .npy encontrado em: {arrays_dir}")

    def __len__(self) -> int:
        return len(self.array_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # 1. Carrega o array de refletividade (H, W) normalizado [0.0, 1.0]
        array_path = self.array_files[idx]
        grid = np.load(array_path).astype(np.float32)
        h, w = grid.shape

        # Adiciona dimensão de canal -> shape: (C, H, W) = (1, H, W)
        image_tensor = torch.from_numpy(grid).unsqueeze(0)

        # 2. Localiza o arquivo de labels correspondente
        base_name = os.path.splitext(os.path.basename(array_path))[0]
        txt_path = os.path.join(self.labels_dir, f"{base_name}.txt")

        boxes = []
        labels = []

        if os.path.exists(txt_path):
            with open(txt_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    
                    # YOLO: class_id, xc, yc, box_w, box_h (normalizados)
                    class_id = int(parts[0])
                    xc, yc, bw, bh = map(float, parts[1:5])

                    # Converte para coordenadas absolutas em pixels [xmin, ymin, xmax, ymax]
                    xmin = (xc - (bw / 2.0)) * w
                    ymin = (yc - (bh / 2.0)) * h
                    xmax = (xc + (bw / 2.0)) * w
                    ymax = (yc + (bh / 2.0)) * h

                    # Clamp para garantir que as caixas não ultrapassem as bordas da matriz
                    xmin = max(0.0, min(xmin, float(w)))
                    ymin = max(0.0, min(ymin, float(h)))
                    xmax = max(0.0, min(xmax, float(w)))
                    ymax = max(0.0, min(ymax, float(h)))

                    # Filtra possíveis caixas degeneradas com área <= 0
                    if (xmax - xmin) > 0 and (ymax - ymin) > 0:
                        boxes.append([xmin, ymin, xmax, ymax])
                        # Em modelos do torchvision: 0 costuma ser background; classes começam em 1
                        labels.append(class_id + 1)

        # 3. Conversão para Tensores do PyTorch
        if len(boxes) > 0:
            boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32)
            labels_tensor = torch.as_tensor(labels, dtype=torch.int64)
            area_tensor = (boxes_tensor[:, 3] - boxes_tensor[:, 1]) * (boxes_tensor[:, 2] - boxes_tensor[:, 0])
            iscrowd_tensor = torch.zeros((len(boxes),), dtype=torch.int64)
        else:
            # Garante integridade se o frame não tiver nenhuma tempestade severa
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)
            area_tensor = torch.zeros((0,), dtype=torch.float32)
            iscrowd_tensor = torch.zeros((0,), dtype=torch.int64)

        # Dicionário de anotações no formato padrão COCO / Torchvision
        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": torch.tensor([idx]),
            "area": area_tensor,
            "iscrowd": iscrowd_tensor
        }

        # Aplica transformações adicionais (ex.: Albumentations / custom transforms), se fornecidas
        if self.transform is not None:
            image_tensor, target = self.transform(image_tensor, target)

        return image_tensor, target


def collate_fn_detection(batch: List[Tuple[torch.Tensor, Dict[str, torch.Tensor]]]):
    """
    Função de colagem customizada para Object Detection.
    Necessária pois o número de tempestades varia por frame, 
    impedindo que os targets sejam concatenados num único tensor retangular.
    """
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    
    # Empilha as imagens em um tensor (B, C, H, W)
    images = torch.stack(images, dim=0)
    
    return images, targets

if __name__ == "__main__":

    ARRAYS_DIR = "./data/processed/arrays"
    LABELS_DIR = "./data/processed/labels"

    # Criação do Dataset completo
    dataset = RadarDataset(arrays_dir=ARRAYS_DIR, labels_dir=LABELS_DIR)
    print(f"Total de frames de radar carregados: {len(dataset)}")

    # Split: 80% treino, 20% validação
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset, 
        [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,  # Executa tudo na thread principal
        collate_fn=collate_fn_detection
    )

    # Inspeção do primeiro batch
    for images, targets in train_loader:
        print(f"\nBatch de Imagens shape: {images.shape} (Batch, Canais, H, W)")
        print(f"Tamanho do lote: {len(targets)} anotações correspondentes.")
        
        for i, target in enumerate(targets):
            print(f"  Frame {i} -> {target['boxes'].shape[0]} células detectadas.")
            if target['boxes'].shape[0] > 0:
                print(f"    Primeira caixa [xmin, ymin, xmax, ymax]: {target['boxes'][0].numpy()}")
                print(f"    Classe: {target['labels'][0].item()}")
        break