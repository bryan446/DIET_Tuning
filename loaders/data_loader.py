"""Dataset utilities for DIET finetuning framework."""

import numpy as np
import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from PIL import Image
from tqdm import tqdm
from torchvision.transforms import v2
from config.data import DATASET_STATS

# HuggingFace datasets
from datasets import load_dataset

# MedMNIST datasets
import medmnist
from medmnist import INFO


class TransformDataset(Dataset):
    """Wrapper to apply transforms to any dataset."""

    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        if isinstance(item, tuple) and len(item) >= 2:
            x, y = item[0], item[1]
        else:
            x = item
            y = torch.tensor(0)

        if self.transform:
            x = self.transform(x)

        return x, y


class DatasetWithIndices(Dataset):
    """Wrapper that assigns diet classes based on sample indices."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, n):
        if isinstance(n, torch.Tensor):
            n = int(n.item())

        item = self.dataset[n]

        if isinstance(item, tuple) and len(item) >= 2:
            x, y = item[0], item[1]
        else:
            x = item
            y = torch.tensor(0)

        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.long)

        if y.dim() == 0:
            y = y.view(1)

        diet_class = torch.tensor(n, dtype=torch.long).view(1)
        return x, y, diet_class

    def __len__(self):
        return int(len(self.dataset))


class HFImageDataset(Dataset):
    """Wrapper for HuggingFace datasets."""

    def __init__(self, hf_dataset, transform=None, input_size=224):
        self.dataset = hf_dataset
        self.transform = transform
        self.resize = transforms.Resize((input_size, input_size))

    def __len__(self):
        return int(len(self.dataset))

    def __getitem__(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = int(idx.item())

        item = self.dataset[idx]

        # Handle both tuple and dictionary formats
        if isinstance(item, tuple):
            image, label = item[0], item[1]
        else:
            image = item["image"]
            label = item.get("label", item.get("labels"))

        if not isinstance(image, Image.Image):
            try:
                image = Image.fromarray(image)
            except Exception:
                pass

        image = self.resize(image)
        if self.transform:
            image = self.transform(image)

        return image, label


class RobustGalaxyDataset(Dataset):
    """Robust dataset class for Galaxy datasets."""

    def __init__(self, hf_dataset, transform=None, limit_samples=None):
        self.dataset = hf_dataset
        self.transform = transform
        self.resize = transforms.Resize((256, 256))

        if limit_samples is not None and limit_samples < len(hf_dataset):
            indices = torch.randperm(len(hf_dataset))[:limit_samples].tolist()
            self.indices = indices
        else:
            self.indices = list(range(len(hf_dataset)))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        original_idx = self.indices[idx]
        sample = self.dataset[original_idx]

        # Handle both tuple and dictionary formats
        if isinstance(sample, tuple):
            image, label = sample[0], sample[1]
        else:
            image = sample["image"]
            label = sample["label"]

        if not isinstance(label, torch.Tensor):
            label = torch.tensor([label], dtype=torch.long)
        elif label.dim() == 0:
            label = torch.tensor([label.item()], dtype=torch.long)

        if not isinstance(image, Image.Image):
            try:
                image = Image.fromarray(image)
            except:
                pass

        image = self.resize(image)
        if self.transform:
            image = self.transform(image)

        diet_class = torch.tensor([idx], dtype=torch.long)
        return image, label, diet_class


def get_dataset(dataset_name="cifar10", root="./data"):
    """Load dataset and return train, validation, test splits.

    Dataset Sources:
    - Torchvision: cifar10, cifar100 (primary)
    - HuggingFace: cifar100 (fallback), food101, fgvc_aircraft, plantnet300k, galaxy10_decals, crop14_balance
    - MedMNIST: All medical datasets (dermamnist, pathmnist, etc.)
    """

    dataset_stats = DATASET_STATS

    if dataset_name.lower() == "cifar10":
        train_dataset = datasets.CIFAR10(root=root, train=True, download=True)
        test_dataset = datasets.CIFAR10(root=root, train=False, download=True)
        train_size = len(train_dataset) // 2
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
        num_classes = 10

    elif dataset_name.lower() == "cifar100":
        # Try torchvision first, fallback to HuggingFace
        try:
            train_dataset = datasets.CIFAR100(root=root, train=True, download=True)
            test_dataset = datasets.CIFAR100(root=root, train=False, download=True)
            train_size = len(train_dataset) // 2
            val_size = len(train_dataset) - train_size
            train_dataset, val_dataset = random_split(
                train_dataset, [train_size, val_size]
            )
            num_classes = 100
        except Exception:
            # Fallback to HuggingFace dataset
            dataset = load_dataset(
                "randall-lab/cifar100", trust_remote_code=True, cache_dir=root
            )
            num_classes = 100
            input_size = dataset_stats["cifar100"]["input_size"]
            train_dataset_full = HFImageDataset(dataset["train"], input_size=input_size)
            test_dataset = HFImageDataset(dataset["test"], input_size=input_size)
            train_size = len(train_dataset_full) // 2
            val_size = len(train_dataset_full) - train_size
            train_dataset, val_dataset = random_split(
                train_dataset_full, [train_size, val_size]
            )

    elif dataset_name.lower() == "food101":
        # HuggingFace dataset with torchvision fallback
        try:
            dataset = load_dataset(
                "randall-lab/food101", trust_remote_code=True, cache_dir=root
            )
            num_classes = 101
            input_size = dataset_stats["food101"]["input_size"]
            train_dataset_full = HFImageDataset(dataset["train"], input_size=input_size)
            test_dataset = HFImageDataset(dataset["test"], input_size=input_size)
            train_size = len(train_dataset_full) // 2
            val_size = len(train_dataset_full) - train_size
            train_dataset, val_dataset = random_split(
                train_dataset_full, [train_size, val_size]
            )
        except Exception:
            train_dataset = datasets.CIFAR100(root=root, train=True, download=True)
            test_dataset = datasets.CIFAR100(root=root, train=False, download=True)
            train_size = len(train_dataset) // 2
            val_size = len(train_dataset) - train_size
            train_dataset, val_dataset = random_split(
                train_dataset, [train_size, val_size]
            )
            num_classes = 100

    elif dataset_name.lower() == "fgvc_aircraft":
        # HuggingFace dataset
        dataset = load_dataset(
            "randall-lab/fgvc-aircraft", trust_remote_code=True, cache_dir=root
        )
        num_classes = 102
        input_size = dataset_stats["fgvc_aircraft"]["input_size"]
        train_dataset = HFImageDataset(dataset["train"], input_size=input_size)
        val_dataset = HFImageDataset(dataset["validation"], input_size=input_size)
        test_dataset = HFImageDataset(dataset["test"], input_size=input_size)

    elif dataset_name.lower() in INFO.keys():
        # MedMNIST dataset
        data_flag = dataset_name.lower()
        info = INFO[data_flag]
        DataClass = getattr(medmnist, info["python_class"])
        num_classes = len(info["label"])
        dataset_config = dataset_stats.get(data_flag, {})
        input_size = dataset_config.get("input_size", 224)
        train_dataset = DataClass(
            split="train", download=True, root=root, size=input_size
        )
        val_dataset = DataClass(split="val", download=True, root=root, size=input_size)
        test_dataset = DataClass(
            split="test", download=True, root=root, size=input_size
        )

    elif dataset_name.lower() == "plantnet300k":
        # HuggingFace dataset
        dataset = load_dataset("mikehemberger/plantnet300K", cache_dir=root)
        num_classes = 85
        input_size = dataset_stats["plantnet300k"]["input_size"]
        train_dataset_full = HFImageDataset(dataset["train"], input_size=input_size)
        if "validation" in dataset:
            test_dataset = HFImageDataset(dataset["validation"], input_size=input_size)
        else:
            test_dataset = HFImageDataset(dataset["test"], input_size=input_size)
        train_size = len(train_dataset_full) // 2
        val_size = len(train_dataset_full) - train_size
        train_dataset, val_dataset = random_split(
            train_dataset_full, [train_size, val_size]
        )

    elif dataset_name.lower() == "galaxy10_decals":
        # HuggingFace dataset
        dataset = load_dataset("matthieulel/galaxy10_decals", cache_dir=root)
        num_classes = 10
        input_size = dataset_stats["galaxy10_decals"]["input_size"]
        train_dataset_full = HFImageDataset(dataset["train"], input_size=input_size)
        test_dataset = HFImageDataset(dataset["test"], input_size=input_size)
        train_size = len(train_dataset_full) // 2
        val_size = len(train_dataset_full) - train_size
        train_dataset, val_dataset = random_split(
            train_dataset_full, [train_size, val_size]
        )

    elif dataset_name.lower() == "crop14_balance":
        # HuggingFace dataset
        dataset = load_dataset("gary109/crop14_balance", cache_dir=root)
        train_dataset_hf = dataset["train"]
        val_dataset_hf = dataset["validation"]
        val_size = len(val_dataset_hf)
        test_size = val_size // 2
        val_indices = list(range(val_size))
        test_indices = val_indices[:test_size]
        val_indices = val_indices[test_size:]
        val_subset = Subset(val_dataset_hf, val_indices)
        test_subset = Subset(val_dataset_hf, test_indices)
        num_classes = 14
        input_size = dataset_stats["crop14_balance"]["input_size"]
        train_dataset = HFImageDataset(
            train_dataset_hf, transform=None, input_size=input_size
        )
        val_dataset = HFImageDataset(val_subset, transform=None, input_size=input_size)
        test_dataset = HFImageDataset(
            test_subset, transform=None, input_size=input_size
        )

    else:
        raise ValueError(
            f"Dataset {dataset_name} not supported or MedMNIST not installed"
        )

    stats = dataset_stats.get(
        dataset_name.lower(),
        {"mean": (0.5,), "std": (0.5,), "input_size": 28, "is_rgb": False},
    )

    return (
        train_dataset,
        val_dataset,
        test_dataset,
        num_classes,
        stats["input_size"],
        stats["mean"],
        stats["std"],
        stats["is_rgb"],
    )


def calculate_dataset_stats(dataset, batch_size=64, max_samples=10000):
    """Calculate mean and std for dataset."""
    from torch.utils.data import DataLoader, Subset
    import random

    if hasattr(dataset, "__len__") and len(dataset) > max_samples:
        indices = random.sample(range(len(dataset)), max_samples)
        dataset_subset = Subset(dataset, indices)
    else:
        dataset_subset = dataset

    if hasattr(dataset, "transform"):
        original_transform = dataset.transform
        dataset.transform = torchvision.transforms.ToTensor()
    elif hasattr(dataset, "dataset") and hasattr(dataset.dataset, "transform"):
        original_transform = dataset.dataset.transform
        dataset.dataset.transform = torchvision.transforms.ToTensor()
    else:

        class StatsDataset(torch.utils.data.Dataset):
            def __init__(self, original_dataset):
                self.dataset = original_dataset
                self.transform = torchvision.transforms.ToTensor()

            def __len__(self):
                return len(self.dataset)

            def __getitem__(self, idx):
                if hasattr(self.dataset, "__getitem__"):
                    item = self.dataset[idx]
                    if isinstance(item, tuple) and len(item) >= 2:
                        img, label = item[0], item[1]
                    else:
                        img = item["image"]
                        label = item.get("label", 0)
                else:
                    raise ValueError("Dataset structure not supported")

                if self.transform:
                    img = self.transform(img)
                return img, label

        dataset_subset = StatsDataset(dataset_subset)
        original_transform = None

    loader = DataLoader(
        dataset_subset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0

    for data, _ in tqdm(loader, desc="Calculating dataset statistics"):
        if not isinstance(data, torch.Tensor):
            continue

        if data.dim() == 3:
            data = data.unsqueeze(1)

        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data**2, dim=[0, 2, 3])
        num_batches += 1

    if hasattr(dataset, "transform") and original_transform is not None:
        dataset.transform = original_transform
    elif (
        hasattr(dataset, "dataset")
        and hasattr(dataset.dataset, "transform")
        and original_transform is not None
    ):
        dataset.dataset.transform = original_transform

    mean = channels_sum / num_batches
    std = (channels_squared_sum / num_batches - mean**2) ** 0.5
    return mean.tolist(), std.tolist()


def create_transforms(mean, std, da_strength=1):
    """Create training and testing transforms."""
    standard_size = 224

    base_transforms = [
        v2.RGB(),
        transforms.Resize((standard_size, standard_size), antialias=True),
    ]

    test_transform = transforms.Compose(
        base_transforms + [transforms.ToTensor(), transforms.Normalize(mean, std)]
    )

    if da_strength > 0:
        aug_list = base_transforms + [
            transforms.RandomResizedCrop(standard_size, antialias=True),
            transforms.RandomHorizontalFlip(),
        ]

        if da_strength > 1:
            aug_list.extend(
                [
                    transforms.RandomApply(
                        [transforms.ColorJitter(0.4, 0.4, 0.4, 0.2)], p=0.3
                    ),
                    transforms.RandomGrayscale(p=0.2),
                ]
            )

        if da_strength > 2:
            aug_list.append(
                transforms.RandomApply(
                    [transforms.GaussianBlur((3, 3), (1.0, 2.0))], p=0.2
                )
            )

        aug_list.extend([transforms.ToTensor(), transforms.Normalize(mean, std)])

        if da_strength > 2:
            aug_list.append(transforms.RandomErasing(p=0.25))

        train_transform = transforms.Compose(aug_list)
    else:
        train_transform = test_transform

    return train_transform, test_transform


def prepare_data_loaders(
    dataset_name,
    batch_size,
    da_strength=1,
    limit_data=np.inf,
    root="./data",
):
    """Prepare data loaders for DIET finetuning"""
    print(f"Loading {dataset_name} dataset...")

    # Get dataset
    (
        training_data_raw,
        val_data_raw,
        test_data_raw,
        num_classes,
        input_size,
        mean,
        std,
        is_rgb,
    ) = get_dataset(dataset_name, root)
    print(
        f"Dataset loaded: input_size={input_size}, mean={mean}, std={std}, "
        f"is_rgb={is_rgb}"
    )

    # Create transforms
    train_transform, test_transform = create_transforms(mean, std, da_strength)

    # Apply transforms to the datasets
    if dataset_name.lower() in ["plantnet300k", "galaxy10_decals"]:
        # For HuggingFace datasets we need to handle the custom dataset wrapper
        training_data = training_data_raw  # No deepcopy needed
        val_data = val_data_raw
        test_data = test_data_raw
        if hasattr(training_data, "transform"):
            training_data.transform = train_transform
            val_data.transform = test_transform
            test_data.transform = test_transform
        else:
            print(
                f"Note: {dataset_name} dataset structure is using "
                f"custom transform handling"
            )
    else:
        # Standard datasets
        import copy

        training_data = copy.deepcopy(training_data_raw)
        val_data = copy.deepcopy(val_data_raw)
        test_data = copy.deepcopy(test_data_raw)

    # Limit training data if specified (before applying transforms)
    if limit_data < np.inf and limit_data < len(training_data):
        print(
            f"Limiting training data to {limit_data} samples "
            f"(out of {len(training_data)})"
        )
        indices = torch.randperm(len(training_data))[:limit_data].tolist()
        training_data = Subset(training_data, indices)
    else:
        print(f"Using full training set: {len(training_data)} samples")

    # Apply transforms after subsetting for standard datasets
    if dataset_name.lower() not in ["plantnet300k", "galaxy10_decals"]:
        # Always use TransformDataset wrapper for consistency
        if hasattr(training_data, "dataset"):
            # This is a Subset, wrap the whole thing with TransformDataset
            training_data = TransformDataset(training_data, train_transform)
        else:
            # This is a regular dataset, wrap it
            training_data = TransformDataset(training_data, train_transform)

        # For val and test data, use TransformDataset wrapper
        # to ensure transforms work
        val_data = TransformDataset(val_data, test_transform)
        test_data = TransformDataset(test_data, test_transform)

    # Special handling for Galaxy dataset
    if dataset_name.lower() == "galaxy10_decals":
        print("\n===== REBUILDING GALAXY DATASET FROM SCRATCH =====")

        # Get the raw dataset again
        raw_dataset = load_dataset("matthieulel/galaxy10_decals")
        train_data = raw_dataset["train"]
        test_data = raw_dataset["test"]

        # Create robust galaxy dataset
        print("Creating robust galaxy dataset...")
        training_data = RobustGalaxyDataset(
            train_data,
            transform=train_transform,
            limit_samples=limit_data if limit_data < np.inf else None,
        )

        val_data = RobustGalaxyDataset(val_data_raw, transform=test_transform)
        test_data = RobustGalaxyDataset(test_data_raw, transform=test_transform)

        print(
            f"Created robust datasets: {len(training_data)} training, "
            f"{len(val_data)} validation, {len(test_data)} test"
        )
        print("===== GALAXY DATASET REBUILDING COMPLETE =====\n")
    else:
        # For non-Galaxy datasets, use the regular DatasetWithIndices wrapper
        training_data = DatasetWithIndices(training_data)

    print(f"Validation set size: {len(val_data)} samples")
    print(f"Test set size: {len(test_data)} samples")

    # Create data loaders
    training_loader = DataLoader(
        training_data,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=16,
    )

    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=16,
    )

    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=16,
    )

    dataset_info = {
        "num_classes": num_classes,
        "num_diet_classes": len(training_data),
        "input_size": input_size,
        "mean": mean,
        "std": std,
        "is_rgb": is_rgb,
        "train_size": len(training_data),
        "val_size": len(val_data),
        "test_size": len(test_data),
    }

    return training_loader, val_loader, test_loader, dataset_info
