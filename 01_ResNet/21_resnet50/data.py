#按固定清单读取图片，进行预处理，组成批次

import csv
import json 
from pathlib import Path,PurePosixPath


from PIL import Image 
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms

# 按照清单读取样本的数据集
class ManifestDataset(Dataset):
    def __init__(self,data_root,manifest_path,classes_path,transform=None):
        self.data_root=Path(data_root)   
        self.transform=transform
        class_info=json.loads(Path(classes_path).read_text(encoding='utf-8'))
        self.classes=class_info['classes']

        class_to_index={
            row['synset']:row['class_index'] for row in self.classes
        }
        self.samples=[]

        with Path(manifest_path).open(encoding='utf-8-sig',newline='')as file:
            reader=csv.DictReader(file)

            for row in reader:
                relative=PurePosixPath(row['relative_path'])
                class_index=int(row['class_index'])
                if (relative.is_absolute()
                   or'..'in relative.parts
                   or'\\'in row['relative_path']
                   or len(relative.parts)!=3
                   or relative.parts[0]not in {'train','val'} 
                   or relative.parts[1]!=row['synset']
                   or class_to_index.get(row['synset'])!=class_index
                ):
                    raise ValueError('清单中的图片路径或标签不正确。')

                self.samples.append(
                    (row['relative_path'],class_index)
                )
            if not self.samples:
                raise ValueError('图片清单为空。')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self,index):
        relative_path,label=self.samples[index]
        with Image.open(self.data_root/relative_path)as source:
            image=source.convert('RGB')
        if self.transform is not None:
            image=self.transform(image)
        return image,label

def build_transform(is_training):
    if is_training:
        operations=[
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(p=0.5)
        ]
    else:
        operations=[
            transforms.Resize(256),
            transforms.CenterCrop(224)
        ]
    operations.extend([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return transforms.Compose(operations)

def create_data_loaders(
        data_root,
        manifest_dir,
        classes_path,
        batch_size=128,
        num_workers=4
):
    train_dataset=ManifestDataset(
        data_root,
        Path(manifest_dir)/'train.csv',
        classes_path,
        build_transform(True)
    )
    val_dataset=ManifestDataset(data_root,Path(manifest_dir)/'val.csv',classes_path,build_transform(False))
    train_loader=DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True
    )
    val_loader=DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False
    )

    return train_loader,val_loader
