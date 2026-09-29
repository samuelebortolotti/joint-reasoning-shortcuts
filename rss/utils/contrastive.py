import torch
import torchvision.transforms as T

AUGMENT = 0

import torch
import torchvision.transforms as T

AUGMENT = 0

def apply_augmentation(batch, dataset="mnist"):
    """
    Apply augmentation to a batch of images.

    Args:
        batch (torch.Tensor): A batch of images, shape (B, 4, C, H, W).
        dataset (str): The dataset type, either 'mnist' or 'clevr'.
        
    Returns:
        torch.Tensor: Augmented batch of images.
    """
    global AUGMENT

    if dataset not in ["mnist", "clevr", "cub"]:
        raise ValueError("Dataset must be 'mnist' or 'clevr' or 'cub.")

    random_crop = T.Compose([
        T.RandomCrop(size=(20, 20)) if dataset == "mnist" else T.RandomCrop(size=(125, 125)),
        T.Resize(size=(28, 28)) if dataset == "mnist" else T.Resize(size=(128, 128))
    ])
    color_distortion = T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1)
    gaussian_blur = T.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0))
    
    # Select augmentation
    if AUGMENT == 0:
        transform = random_crop
    elif AUGMENT == 1:
        transform = color_distortion
    elif AUGMENT == 2:
        transform = gaussian_blur
    else:
        raise ValueError("Invalid augmentation type.")

    AUGMENT = (AUGMENT + 1) % 3  # Cycle through augmentations

    if dataset == "mnist":
        digit1 = torch.stack([transform(img) for img in batch[:, :, :, :28]])
        digit2 = torch.stack([transform(img) for img in batch[:, :, :, 28:]])

        return torch.cat([digit1, digit2], dim=-1)
    else:
        B, num_images, C, H, W = batch.shape
        augmented_batch = torch.clone(batch)

        for b in range(B):
            for i in range(num_images):
                if torch.all(batch[b, i, :, :] == -1):  # Skip if the image is full of -1
                    continue
                augmented_batch[b, i, :, :] = transform(batch[b, i, :, :])

        return augmented_batch
