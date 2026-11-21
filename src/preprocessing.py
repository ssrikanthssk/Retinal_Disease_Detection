
from PIL import Image, ImageOps
from torchvision import transforms
from torchvision.transforms import InterpolationMode

IMAGE_SIZE = 224

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class ResizeWithPadding:
    """Resize while preserving the original aspect ratio."""

    def __init__(self, size=224):
        self.size = size

    def __call__(self, image):
        image = image.convert("RGB")

        # Resize proportionally to fit inside the target size
        image = ImageOps.contain(
            image,
            (self.size, self.size),
            method=Image.Resampling.BILINEAR
        )

        # Pad remaining space with black
        width, height = image.size

        left = (self.size - width) // 2
        top = (self.size - height) // 2
        right = self.size - width - left
        bottom = self.size - height - top

        return ImageOps.expand(
            image,
            border=(left, top, right, bottom),
            fill=(0, 0, 0)
        )



def get_train_transforms():
    return transforms.Compose([
        ResizeWithPadding(IMAGE_SIZE),

        transforms.ColorJitter(
            brightness=0.08,
            contrast=0.08
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD
        )
    ])


def get_val_transforms():
    return transforms.Compose([
        ResizeWithPadding(IMAGE_SIZE),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD
        )
    ])
