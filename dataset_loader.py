"""
dataset_loader.py - Carga y preparación del dataset para entrenamiento de IA
Estructura esperada:
dataset/
├── train/
│   ├── desplazamiento/
│   ├── normal/
│   ├── spaghetti/
│   └── warping/
└── val/
    ├── desplazamiento/
    ├── normal/
    ├── spaghetti/
    └── warping/
"""

import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_data_loaders(data_dir="dataset", batch_size=32, num_workers=0):
    """
    Crea los DataLoaders para entrenamiento y validación
    
    Args:
        data_dir: directorio raíz del dataset
        batch_size: tamaño del lote
        num_workers: número de workers para carga de datos
    
    Returns:
        train_loader: DataLoader de entrenamiento
        val_loader: DataLoader de validación
        class_names: lista con nombres de las clases
    """
    
    # Transformaciones para entrenamiento (con aumentación de datos)
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    # Transformaciones para validación (sin aumentación)
    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    # Cargar datasets
    train_dataset = datasets.ImageFolder(
        os.path.join(data_dir, "train"),
        transform=train_transforms
    )

    val_dataset = datasets.ImageFolder(
        os.path.join(data_dir, "val"),
        transform=val_transforms
    )

    # Crear DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    print(f"✅ Dataset cargado correctamente")
    print(f"   Clases: {train_dataset.classes}")
    print(f"   Train: {len(train_dataset)} imágenes")
    print(f"   Val: {len(val_dataset)} imágenes")

    return train_loader, val_loader, train_dataset.classes

def get_test_loader(data_dir="dataset", batch_size=32):
    """
    Carga solo el conjunto de prueba/validación
    
    Args:
        data_dir: directorio raíz del dataset
        batch_size: tamaño del lote
    
    Returns:
        test_loader: DataLoader de prueba
        class_names: lista con nombres de las clases
    """
    
    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    test_dataset = datasets.ImageFolder(
        os.path.join(data_dir, "val"),
        transform=test_transforms
    )

    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False
    )

    return test_loader, test_dataset.classes