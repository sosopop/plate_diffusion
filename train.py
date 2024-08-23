import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import glob
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from multiprocessing import freeze_support
from diffusion__model import DiffusionModel
import utils
from license_plate_dataset import LicensePlateDataset
from license_plate_vocab import LicensePlateVocab
import matplotlib.font_manager as fm
from torch.nn import functional as F

# 设置中文字体
font_path = "C:/Windows/Fonts/simhei.ttf"  # 可以选择其他中文字体
prop = fm.FontProperties(fname=font_path)


def train(model, train_loader, optimizer, device, epoch):
    model.train()
    loss_total = 0
    total_count = 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}", unit="batch") 
    for batch, labels in pbar:
        batch = batch.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        
        t = torch.randint(0, model.num_timesteps, (batch.size(0),), device=device)
        x_t, noise = model.q_sample(batch, t)
        predicted_noise = model(x_t, t.float() / model.num_timesteps, labels)
        
        loss = F.mse_loss(noise, predicted_noise)
        loss.backward()
        loss_total += loss.item() * batch.size(0)
        total_count += batch.size(0)
        optimizer.step()
        
        pbar.set_postfix(loss=loss.item())
        
        # if pbar.n > 100:
        #     break
    
    return loss_total / total_count


def inference(model, dataloader, device, filename):
    model.eval()
    # Get a batch of images and labels
    images, labels = next(iter(dataloader))
    images = images.to(device)
    labels = labels.to(device)
    
    # Select 8 images for visualization
    n_images = 4
    selected_images = images[:n_images]
    selected_labels = labels[:n_images]
    
    reconstructed_images = []
    with torch.no_grad():
        for t in [1, 10, 100, 200, 500, 999]:  # Different noise levels
            # Forward diffusion
            x_t, _ = model.q_sample(selected_images, torch.full((n_images,), t, device=device))
            reconstructed_images.append(x_t.cpu())
            
            # Backward diffusion
            x_recon = x_t
            for t_prime in reversed(range(t + 1)):
                t_batch = torch.full((n_images,), t_prime, device=device, dtype=torch.long)
                x_recon = model.p_sample(x_recon, t_batch, selected_labels)
            
            reconstructed_images.append(x_recon.cpu())
    
    # Plotting
    fig, axes = plt.subplots(n_images, 13, figsize=(40, 3 * n_images))
    for i in range(n_images):
        axes[i, 0].imshow(selected_images[i].permute(1, 2, 0).clip(0, 1).cpu(), vmin=0, vmax=1)
        axes[i, 0].set_title(f'Original (Label: {LicensePlateVocab.sequence_to_text(selected_labels[i].tolist())})')
        axes[i, 0].axis('off')
        for j, t in enumerate([1, 10, 100, 200, 500, 999]):
            axes[i, j*2+1].imshow(reconstructed_images[j*2][i].permute(1, 2, 0).clip(0, 1), vmin=0, vmax=1)
            axes[i, j*2+1].set_title(f'x_t={t}')
            axes[i, j*2+1].axis('off')
            axes[i, j*2+2].imshow(reconstructed_images[j*2+1][i].permute(1, 2, 0).clip(0, 1), vmin=0, vmax=1)
            axes[i, j*2+2].set_title(f'r_t={t}')
            axes[i, j*2+2].axis('off')
    
    plt.tight_layout()
    os.makedirs('samples', exist_ok=True)
    plt.savefig(os.path.join('samples', filename))
    plt.close()


# 新增：获取最新的checkpoint文件
def get_latest_checkpoint(checkpoint_dir):
    checkpoints = glob.glob(os.path.join(checkpoint_dir, 'epoch_*.pth'))
    if not checkpoints:
        return None
    return max(checkpoints, key=os.path.getctime)

def load_dict_from_checkpoint(dict, model):
    model_state_dict = model.state_dict()
    for name, param in dict.items():
        if name in model_state_dict:
            if model_state_dict[name].size() == param.size():
                model_state_dict[name].copy_(param)
            else:
                print(f"Skipping layer: {name} due to size mismatch ({model_state_dict[name].size()} vs {param.size()})")
        else:
            print(f"Skipping layer: {name} as it is not in the model")

# 主函数
def main():
    freeze_support()
    # 设置参数
    epochs = 1000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载数据
    transform = transforms.Compose([
        transforms.Resize(128),
        transforms.ToTensor(),
    ])
    
    train_dataset = LicensePlateDataset(r'D:\code\plate_gen\data\plate\train', transform=transform)
    test_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
    
    # 初始化模型
    diffusion_model = DiffusionModel(num_timesteps=1000)
    diffusion_model.to(device)
    optimizer = torch.optim.Adam(diffusion_model.parameters(), lr=0.0001)
    
    # 确保存在保存checkpoints的目录
    checkpoint_dir = 'checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 尝试加载最新的checkpoint
    start_epoch = 1
    latest_checkpoint = get_latest_checkpoint(checkpoint_dir)
    if latest_checkpoint:
        # latest_checkpoint = "checkpoints/gan_epoch_4.pth"
        checkpoint = torch.load(latest_checkpoint)
        if 'diffusion_model' in checkpoint:
            load_dict_from_checkpoint(checkpoint['diffusion_model'], diffusion_model)
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
        
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming from epoch {start_epoch}, loaded checkpoint: {latest_checkpoint}")
    else:
        print("No checkpoint found, starting from scratch.")

    writer = SummaryWriter(log_dir='runs/training')

    inference(diffusion_model, test_loader, device, f"reconstruction_last.png")
    
    # 训练模型
    for epoch in range(start_epoch, epochs + 1):
        loss = train(diffusion_model, train_loader, optimizer, device, epoch)
        print(f'Epoch {epoch}, Loss: {loss:.4f}')
        
        # 保存checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f'epoch_{epoch}.pth')
        torch.save({
            'epoch': epoch,
            'diffusion_model': diffusion_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'loss': loss,
        }, checkpoint_path)
        print(f'Checkpoint saved to {checkpoint_path}')
        
        writer.add_scalar('Loss', loss, epoch)
        
        # 进行推理并保存图像
        inference(diffusion_model, test_loader, device, f"reconstruction_epoch_{epoch+1}.png")
        print(f'Inference image saved for epoch {epoch}')

if __name__ == '__main__':
    main()