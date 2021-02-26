from torch.utils import data 
import numpy as np 

from PIL import Image 

class DetectionDataset(data.Dataset):
    def __init__(self, data_dir, parser = None, transform = None):
        super(DetectionDataset, self).__init__() 
        self.data_dir = data_dir
        self.parser = parser 
        self.transform = transform 

        pass 
    def __getitem__(self, index):
        """
            Args: 
                index (int): Index 
            Returns:
                tuple: Tuple (image, annotations (target))
        """
        img_info = self.parser.img_infos[index]
        target = dict(img_idx=index, img_size=(img_info['width'], img_info['height']))
        if self.parser.has_labels:
            ann = self.parser.get_ann_info(index) 
            target.update(ann) 
        img_path = self.data_dir / img_info['file_name']
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img, target = self.transform(img, target)
        return super().__getitem__(index)

    def __len__(self):
        return len(self.parser.img_ids)