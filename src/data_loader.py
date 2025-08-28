import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
import os
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

def get_data_loaders(data_dir, batch_size=32, img_size=224, num_workers=2, visualize=False):
    print("📁 Initializing data loaders...")

    # ImageNet normalization
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # Paths
    train_path = os.path.join(data_dir, 'train')
    val_path   = os.path.join(data_dir, 'val')
    test_path  = os.path.join(data_dir, 'test')

    # Datasets
    train_dataset = datasets.ImageFolder(train_path, transform=train_transform)
    val_dataset   = datasets.ImageFolder(val_path, transform=test_transform)
    test_dataset  = datasets.ImageFolder(test_path, transform=test_transform)

    class_names = train_dataset.classes
    class_to_idx = train_dataset.class_to_idx
    print(f"🧪 Classes ({len(class_names)}): {class_names}")
    print(f"🔢 Class-to-index mapping: {class_to_idx}")

    # Class distribution
    def get_class_distribution(dataset):
        targets = [label for _, label in dataset.samples]
        counts = Counter(targets)
        return {class_names[k]: v for k, v in counts.items()}

    print(f"📊 Train class distribution: {get_class_distribution(train_dataset)}")
    print(f"📊 Val class distribution:   {get_class_distribution(val_dataset)}")
    print(f"📊 Test class distribution:  {get_class_distribution(test_dataset)}")

    # Weighted sampling
    targets = [label for _, label in train_dataset.samples]
    class_counts = Counter(targets)
    class_weights_dict = {cls: 1.0 / count for cls, count in class_counts.items()}
    sample_weights = [class_weights_dict[label] for label in targets]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    print("⚖️ Using WeightedRandomSampler for training.")

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=num_workers)

    # Class weights for loss function
    class_weights = torch.tensor([class_weights_dict[i] for i in range(len(class_names))], dtype=torch.float)

    # Sample inspection
    sample_img, sample_label = train_dataset[0]
    print(f"🖼️ Sample image shape: {sample_img.shape}")
    print(f"🔍 Sample label: {class_names[sample_label]}")

    def imshow(img, title):
        img = img.numpy().transpose((1, 2, 0))
        img = std * img + mean
        img = np.clip(img, 0, 1)
        plt.imshow(img)
        plt.title(title)
        plt.axis('off')
        plt.show()

    if visualize:
        print("📸 Visualizing one image per class...")
        seen_classes = set()
        for path, label in train_dataset.samples:
            class_name = class_names[label]
            if class_name not in seen_classes:
                img = train_dataset.loader(path)
                img = train_transform(img)
                imshow(img, f"Class: {class_name}")
                seen_classes.add(class_name)
            if len(seen_classes) == len(class_names):
                break

    print("✅ Data loaders ready.")
    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
        "class_weights": class_weights,
        "class_names": class_names
    }

# Optional test run
if __name__ == "__main__":
    data_dir = "D:/College/Second Year/EDI/Pneumonia_Detection/DATA"
    loaders = get_data_loaders(data_dir, batch_size=32, img_size=224, visualize=True)

    print(f"📦 Train batches: {len(loaders['train'])}")
    print(f"📦 Validation batches: {len(loaders['val'])}")
    print(f"📦 Test batches: {len(loaders['test'])}")
    print(f"⚖️ Class weights: {loaders['class_weights']}")
    print(f"🧪 Class names: {loaders['class_names']}")