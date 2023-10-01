from torch.utils.data import Dataset
from pathlib import Path
import cv2
import numpy as np
import torchvision.transforms.v2 as T


class S2sDataSet(Dataset):
    def __init__(self, dir_path: Path, size):
        self.source_data = list((dir_path / 'scheme').iterdir())
        self.size = size
        self.transform = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
        ])

    def __getitem__(self, index):
        scheme_path = self.source_data[index]
        simulation_path = scheme_path.parent.parent / 'simulation' / scheme_path.name
        prompt = 'flat design'

        hint = cv2.imread(str(scheme_path))
        target = cv2.imread(str(simulation_path))
        # Do not forget that OpenCV read images in BGR order.
        hint = cv2.cvtColor(hint, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        hint = cv2.resize(hint, self.size)
        target = cv2.resize(target, self.size)

        # Normalize source images to [0, 1].
        hint = hint.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0
        target, hint = self.transform(target, hint)

        return dict(jpg=target, txt=prompt, hint=hint)

    def __len__(self):
        return len(self.source_data)


class PredictS2sDataSet(Dataset):
    def __init__(self, dir_path: Path, size):
        self.source_data = list((dir_path).iterdir())
        self.size = size

    def __getitem__(self, index):
        scheme_path = self.source_data[index]
        prompt = 'flat design'

        source = cv2.imread(str(scheme_path))
        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)

        source = cv2.resize(source, self.size)

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        return dict(txt=prompt, hint=source, jpg=source, filename=scheme_path.name)

    def __len__(self):
        return len(self.source_data)
