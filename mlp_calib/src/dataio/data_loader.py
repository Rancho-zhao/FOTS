"""
  Data loader for the color-normal datasets
"""
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# from src.dataio.digit_dataset import DigitRealImageAnnotDataset
from .digit_dataset import DigitRealImageAnnotDataset

def data_loader(dir_dataset, params):
    """A data loader for the color-normal datasets
    Args:
        dir_dataset: path to the dataset
        params: a dict of parameters
    """
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = DigitRealImageAnnotDataset( dir_dataset=dir_dataset, annot_file=params.annot_file,
                                          transform=transform, annot_flag=params.annot_flag)
    dataloader = DataLoader(dataset, batch_size=params.batch_size, shuffle=params.shuffle,
                            num_workers=params.num_workers)

    return dataloader, dataset
