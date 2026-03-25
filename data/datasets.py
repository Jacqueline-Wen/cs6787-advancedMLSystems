from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def get_mnist(batch_size, data_dir="./data_cache", subset_indices=None):
    """Return MNIST train and test DataLoaders.

    Args:
        batch_size: Mini-batch size.
        data_dir: Where to download/cache the dataset.
        subset_indices: Optional list of indices to select a subset of
            the training set (useful for partitioning across workers later).

    Returns:
        (train_loader, test_loader)
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=transform)

    if subset_indices is not None:
        train_dataset = Subset(train_dataset, subset_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    return train_loader, test_loader


def get_cifar10(batch_size, data_dir="./data_cache", subset_indices=None):
    """Return CIFAR-10 train and test DataLoaders."""
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    train_dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(data_dir, train=False, download=True, transform=transform_test)

    if subset_indices is not None:
        train_dataset = Subset(train_dataset, subset_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    return train_loader, test_loader
