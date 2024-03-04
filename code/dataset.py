import numpy as np
import io
import tarfile
import torch
from torch.utils.data import Dataset
from torchvision import transforms

def update_type_float32(x):
    return x.type(torch.float32)

def get_norm(mean_per_layer, std_per_layer):
    def normalize_data(x):
        return (x - mean_per_layer[np.newaxis, np.newaxis, :]) / std_per_layer[np.newaxis, np.newaxis, :]
    return normalize_data

def custom_collate(batch):
    # Filter out mostly black images based on a threshold
    threshold = 0.1
    filtered_batch = [sample for sample in batch if sample.mean() > threshold]
    samples = [sample for sample in filtered_batch]
    if not samples:
        return None
    stacked_samples = torch.stack(samples, dim=0)
    return stacked_samples


def load_numpy_array_from_tar_gz(tar_gz_file):
    with tarfile.open(tar_gz_file, 'r:gz') as tar:
        numpy_array_file = tar.extractfile('numpy_array.npy')
        numpy_array_bytes = numpy_array_file.read()
        numpy_array = np.load(io.BytesIO(numpy_array_bytes))
    return numpy_array

class SpectralDataset(Dataset):
    def __init__(self, data_path, pad_x, pad_y):
        path = f"{data_path}/data.tar.gz"
        print(f"Loading data from {path}")
        self.data = load_numpy_array_from_tar_gz(path)
        mean_per_layer = np.mean(self.data, axis=(0, 1))
        std_per_layer = np.std(self.data, axis=(0, 1))
        self.pad_x = pad_x
        self.pad_y = pad_y

        self.transform = transforms.Compose([
            get_norm(mean_per_layer, std_per_layer),
            transforms.ToTensor(),
            update_type_float32,
        ])

        if self.data.shape[0] < pad_x or self.data.shape[1] < pad_y:
            raise ValueError(f"Padding should be lower than data shape data.shape = {self.data.shape}")
        
    def __len__(self):
        return (self.data.shape[0] - self.pad_x) * (self.data.shape[1] - self.pad_y)

    def __getitem__(self, index):
        x = index % (self.data.shape[0] - self.pad_x)
        y = index // (self.data.shape[0] - self.pad_x)
        img = self.data[x : x + self.pad_x, y : y + self.pad_y, :]
        transformed_img = self.transform(img).transpose(0, 2)
        return transformed_img

