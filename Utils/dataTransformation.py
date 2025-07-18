from torchvision import transforms


def transformations(mode='train'):
    """
    Returns a torchvision transformation pipeline for the given mode.

    Args:
        mode (str): One of ['train', 'val', 'test']. Controls augmentation behavior.

    Returns:
        torchvision.transforms.Compose: Transform pipeline to apply to each image.
    """
    mean = [0.4489, 0.3227, 0.3442]
    std = [0.2235, 0.1850, 0.1900]

    if mode == 'train':
        return transforms.Compose([
            transforms.Resize((250, 250)),
            transforms.RandomCrop(224),
            transforms.ColorJitter(0.05, 0.05, 0.05, 0.05),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(5),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    elif mode in {'val', 'test'}:
        return transforms.Compose([
            transforms.Resize((250, 250)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        raise ValueError(f"Invalid mode: '{mode}'. Expected 'train', 'val', or 'test'.")