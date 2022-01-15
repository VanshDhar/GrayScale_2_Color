from typing import Tuple
from torch.utils.data import Dataset
import torchvision.transforms as T
import numpy as np
import glob
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
import cv2

class ColorizeData(Dataset):
    def __init__(self,img_dir='/Users/vanshdhar/Desktop/samsung_challenge/landscape_images', train='train',train_val_split=0.15):
        self.dataset = glob.glob(img_dir+'/*.jpg')#os.path.join(path, file)
        self.train=train
        if(train_val_split==0.0):
            self.imgs_path = {}
            self.imgs_path['train'] = self.dataset
            self.imgs_path['val'] = []
        else:
            train_idx, val_idx = train_test_split(list(range(len(self.dataset))), test_size=train_val_split,shuffle=False)
            self.imgs_path = {}
            self.imgs_path['train'] = Subset(self.dataset, train_idx)
            self.imgs_path['val'] = Subset(self.dataset, val_idx)
        self.img_dir = img_dir
        self.input_transform = T.Compose([T.ToTensor(),
                                          T.Resize(size=(256,256)),
                                          T.Grayscale(num_output_channels=1),
                                          T.Normalize((0.5), (0.5))
                                          ])
        # Use this on target images(colorful ones)
        self.target_transform = T.Compose([T.ToTensor(),
                                           T.Resize(size=(256,256)),
                                           T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    def __len__(self) -> int:
        # return Length of dataset
        return len(self.imgs_path[self.train])
    
    def __getitem__(self, index: int) :#-> Tuple(torch.Tensor, torch.Tensor)
        # Return the input tensor and output tensor for training
        label = cv2.imread(self.imgs_path[self.train][index])
        label = np.array(label)
        image = self.input_transform(label)
        label = self.target_transform(label)
        return (image, label)
        
