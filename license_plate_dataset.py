import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from license_plate_vocab import LicensePlateVocab

class LicensePlateDataset(Dataset):
    def __init__(self, image_folder, max_length=16, transform=None):
        self.image_folder = image_folder
        self.image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]
        self.transform = transform
        self.max_length = max_length

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_folder, img_name)
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        # 提取车牌号
        plate_number = img_name.split('-')[0]
        label = LicensePlateVocab.text_to_sequence(plate_number, self.max_length)

        return image, torch.tensor(label, dtype=torch.long)

if __name__ == "__main__":
    # 设置数据变换
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # 最大序列长度
    max_length = 16  # 适当增加以包含EOS和可能的PAD

    # 创建数据集和数据加载器
    train_folder = r'data\plate\train'
    val_folder = r'data\plate\val'

    train_dataset = LicensePlateDataset(train_folder, max_length, transform)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    data_iter = iter(train_loader)
    images, labels = next(data_iter)

    # 设置中文字体
    font_path = "C:/Windows/Fonts/simhei.ttf"  # 可以选择其他中文字体
    prop = fm.FontProperties(fname=font_path)
    plt.rcParams['font.sans-serif'] = [font_path]
    plt.rcParams['axes.unicode_minus'] = False

    # 显示图像和标签
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    for i in range(4):
        for j in range(4):
            img = images[i*4+j].permute(1, 2, 0).numpy()
            label = labels[i*4+j].numpy()
            label_str = LicensePlateVocab.sequence_to_text(label)
            axes[i][j].imshow(img)
            axes[i][j].set_title(label_str, fontproperties=prop)
            axes[i][j].axis('off')

    # 保存图像
    plt.savefig('batch_images.png')
    plt.show()
