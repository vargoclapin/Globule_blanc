from pathlib import Path
from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import download_and_extract_archive, verify_str_arg
from torchvision import transforms
from lightning import LightningDataModule
from torch.utils.data import DataLoader

class BarcelonaDataset(ImageFolder):
    url = "https://cloud.minesparis.psl.eu/index.php/s/Rv8LxYm5rqTKOgS/download"
    filename = "barcelona.zip"
    md5sum = "aad6827e0000988c5e3d72ace95da661"
    foldername = "barcelona"

    def __init__(self, root, split="train", transform=None, target_transform=None, download=False):
        self.split = verify_str_arg(split, "split", ("train", "valid", "test"))

        self.root = Path(root).expanduser()

        if download:
            self._download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        split_dir = self.root / self.foldername / self.split
        super().__init__(root=str(split_dir), transform=transform, target_transform=target_transform)


    def _check_exists(self) -> bool:
        target = self.root / self.foldername
        return target and target.is_dir()

    def _download(self) -> None:
        if not self._check_exists():
            download_and_extract_archive(
                url=self.url,
                download_root=str(self.root),
                filename=self.filename,
                md5=self.md5sum,
                remove_finished=False
            )


class BarcelonaDataModule(LightningDataModule):
    def __init__(self, root, batch_size=32, num_workers=4, pin_memory=True, image_size=(363, 360)):
        super().__init__()
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.image_size = image_size

        self.mean = [0.87450084, 0.74860094, 0.72014712]
        self.std = [0.15908252, 0.18541114, 0.08004474]

        self.transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])

    def prepare_data(self):
        BarcelonaDataset(root=self.root, download=True)

    def setup(self, stage=None):
        self.train_ds = BarcelonaDataset(root=self.root, split="train", transform=self.transform)
        self.val_ds = BarcelonaDataset(root=self.root, split="valid", transform=self.transform)
        self.test_ds = BarcelonaDataset(root=self.root, split="test",  transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=self.pin_memory)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=self.pin_memory)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=self.pin_memory)


