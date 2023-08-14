import os
import torch
from glob import glob
import PIL.Image as pilimage
import PIL.ImageDraw as pildraw
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
from tools import minmax2xywh, xywh2minmax, normalize_target


class KITTI_dataset(Dataset):
    # type, truncated, occluded, alpha, bbox, dimensions, location, rotation_y, score
    def __init__(self, image_path=None, target_path=None, transforms=None) -> None:
        self.classes = ['car', 'van', 'truck', 'pedestrian', 'person_sitting', 'cyclist', 'tram', 'misc']
        self.transforms = transforms

        image_file_paths = glob(os.path.join(image_path, "*.[PpJj]*[NnPp]*[Gg]"))
        target_file_paths = glob(os.path.join(target_path, "*.[Tt]*[Xx]*[Tt]"))
        assert len(image_file_paths) == len(target_file_paths)
        # file_names = [os.path.splitext(os.path.basename(image_file_path))[0] for image_file_path in image_file_paths]
        image_file_paths.sort()
        target_file_paths.sort()
        # print(image_file_paths)

        pbar = tqdm(total=len(image_file_paths), leave=True)
        self.images = []
        self.targets = []
        # print(os.path.splitext(os.path.basename(image_file_paths[0]))[0])
        for image_file_path, target_file_path in zip(image_file_paths, target_file_paths[:32]):
            image_file_name = os.path.splitext(os.path.basename(image_file_path))[0]
            target_file_name = os.path.splitext(os.path.basename(target_file_path))[0]
            assert image_file_name == target_file_name
            # print(image_file_name, target_file_name)

            image = pilimage.open(image_file_path)
            width, height = image.size
            # draw = pildraw.Draw(image)

            target = []
            with open(target_file_path, 'r') as f:
                for line in f.readlines():
                    line = line.strip().split(" ")
                    if line[0] == "DontCare":
                        continue
                    cls, minx, miny, maxx, maxy = line[0], line[4], line[5], line[6], line[7]
                    cls = self.classes.index(cls.lower())
                    cls, minx, miny, maxx, maxy = map(float, [cls, minx, miny, maxx, maxy])
                    # print(cls, minx, miny, maxx, maxy)
                    target += [[cls, minx, miny, maxx, maxy]]
            # assert len(target) != 0
            target = np.array(target)
            # print(target)
            target = minmax2xywh(target)
            target = normalize_target(target, (width, height))
            target = np.concatenate((np.zeros((len(target), 1)), target), 1)
            
            self.images += [image]
            self.targets += [target]
            # print(target)
            pbar.update()
        pbar.close()

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        image = self.images[index]
        target = self.targets[index]
        
        if self.transforms is not None:
            image, target = self.transforms((image, target))
        return image, target

def collate_fn(batch):
    #skip invalid frames
    if len(batch) == 0:
        return
    images, targets = list(zip(*batch))
    images = torch.stack(list(images))

    for i, boxes in enumerate(targets):
        boxes[:, 0] = i
    targets = torch.cat(targets, 0)
    return images, targets



if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from transforms import get_transforms

    train_image_dir = "/Users/shinhyeonjun/code/yolov3_pytorch/data/kitti_dataset/training/images"
    train_target_dir = "/Users/shinhyeonjun/code/yolov3_pytorch/data/kitti_dataset/training/annotations"
    transforms = get_transforms()

    train_dataset = KITTI_dataset(train_image_dir, train_target_dir, transforms=transforms)
    train_lodader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=False, drop_last=True, collate_fn=collate_fn)
    for images, targets in train_lodader:
        print(images.shape, targets)