import torch
from torchvision.transforms import Compose
from torchvision.transforms.functional import resize, to_tensor, normalize, InterpolationMode

class ToTensor:
    def __init__(self) -> None:
        pass
    def __call__(self, data):
        image, target = data
        image = to_tensor(image)
        target = torch.from_numpy(target)
        return image, target

class Resize:
    def __init__(self, tsize:tuple) -> None:
        self.width = tsize[0]
        self.height = tsize[1]

    def __call__(self, data):
        image, target = data
        image = resize(image, (self.height, self.width), InterpolationMode.BILINEAR)
        # target = to_tensor(target)
        return image, target      
    
class Normalize:
    def __init__(self, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)) -> None:
        self.mean = mean
        self.std = std
    def __call__(self, data):
        image, target = data
        image = normalize(image, self.mean, self.std)
        return image, target


def get_transforms():
    transforms = Compose([Resize((416, 416)),
                          ToTensor(),
                          Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
    return transforms