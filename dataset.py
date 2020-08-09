from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
class GAN_Dataset(Dataset):
    def __init__(self,path,dataset):
        self.data=dataset
        self.path=path
    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        data=Image.open(self.path+self.data[item])
        transform = transforms.Compose([
            transforms.Resize(96),
            transforms.CenterCrop(96),
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5, 0.5], [0.5,0.5, 0.5])
        ])
        return transform(data)

