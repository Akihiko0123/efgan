import os
import random
from glob import glob
from warnings import filterwarnings 
import argparse 
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.init as init
import yaml
from natsort import natsorted
from PIL import Image
from skimage import io, transform
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image
import time
#from torchsummary import summary
filterwarnings("ignore")  # warningをオフにする
# 作成したモジュール
from DataLoader import LoadFromFolder
from Encoder import Encoder
from Generator import Generator
from Discriminator import Discriminator

def Anomaly_score(x, E_x, G_E_x, Lambda=0.1):
    _,x_feature = model_D(x, E_x)
    _,G_E_x_feature = model_D(G_E_x, E_x)
    residual_loss = criterion_L1(x, G_E_x)
    discrimination_loss = criterion_L1(x_feature, G_E_x_feature)
    
    total_loss = (1-Lambda)*residual_loss + Lambda*discrimination_loss
    total_loss = total_loss.item()
    
    return total_loss

## 重みの初期化を行う関数
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

parser = argparse.ArgumentParser(description='efficientgan train script')
parser.add_argument('--config_file', type=str, default="efgan_config.yml")
args = parser.parse_args()

if __name__ == '__main__':
    ## 時間測定開始
    st = time.time()

    ## 作業ディレクトリパス
    work_dir = "."
    
    ## コンフィグファイル読み込み
    config_path = os.path.join(os.path.join(work_dir,"config"),args.config_file)
    config_dict = yaml.safe_load(open(config_path, 'r'))

    ## 辞書から変数読み込み
    IMAGE_SIZE = config_dict["image_size"]
    EMBED_SIZE = config_dict["embed_size"]
    BATCH_SIZE = config_dict["batch_size"]
    LR = config_dict["lr"]
    EPOCHS = config_dict["epochs"]
    EPOCHS = int(EPOCHS)
    OUT_MODEL_DIR = config_dict["out_model_dir"]
    OUT_IMAGE_Z_DIR = config_dict["out_image_z_dir"]
    OUT_IMAGE_RECONSTRUCT = config_dict["out_image_reconst"]

    ## 出力フォルダパス
    OUT_MODEL_DIR = os.path.join(work_dir,OUT_MODEL_DIR)
    OUT_IMAGE_Z_DIR = os.path.join(work_dir,OUT_IMAGE_Z_DIR)
    OUT_IMAGE_RECONSTRUCT = os.path.join(work_dir,OUT_IMAGE_RECONSTRUCT)

    # makedirsで複数フォルダ作成
    os.makedirs(OUT_MODEL_DIR,exist_ok=True)
    os.makedirs(OUT_IMAGE_Z_DIR,exist_ok=True)
    os.makedirs(OUT_IMAGE_RECONSTRUCT,exist_ok=True)

    ## データパス
    train_dir = os.path.join(work_dir,"train")
    val_dir = os.path.join(work_dir,"val")

    ## 画像を読み込む際の前処理
    device = "cuda" if torch.cuda.is_available() else "cpu"  # GPUが使えるならGPUで、そうでないならCPUで計算する
    kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}  

    transform_dict = {
        "train": transforms.Compose(
            [
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),  # IMAGE_SIZEにreshape
                transforms.RandomHorizontalFlip(), # ランダムに左右反転を行う
                transforms.ToTensor(), # 0-255 -> 0-1に変換
            ]
        ),
        "test": transforms.Compose(
            [
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),  # IMAGE_SIZEにreshape
                transforms.ToTensor(),
            ]
        ),
    }

    # 読み込み
    train_dataset = LoadFromFolder(train_dir, transform=transform_dict["train"])
    test_dataset = LoadFromFolder(val_dir, transform=transform_dict["test"])
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size = BATCH_SIZE, shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size = BATCH_SIZE, shuffle=True, **kwargs)

    ## モデルの定義
    model_E = Encoder(EMBED_SIZE=EMBED_SIZE).to(device)
    model_E.apply(weights_init)

    model_G = Generator(EMBED_SIZE=EMBED_SIZE).to(device)
    model_G.apply(weights_init)

    model_D = Discriminator(EMBED_SIZE=EMBED_SIZE).to(device)
    model_D.apply(weights_init)

    ## 損失関数と最適化手法
    criterion = nn.BCELoss()  # 評価関数=二値交差検証
    criterion_L1 = nn.L1Loss(reduction="sum") # 異常スコア計測用

    # gとeは共通のoptimizer
    optimizer_ge =  torch.optim.Adam(list(model_G.parameters()) + list(model_E.parameters()), lr=LR, betas=(0.5,0.999))
    optimizer_d = torch.optim.Adam(model_D.parameters(), lr=LR, betas=(0.5,0.999))  # Discriminatorのoptimizer

    # 学習率の減衰：50エポックで0.9倍
    scheduler_ge = torch.optim.lr_scheduler.StepLR(optimizer_ge, step_size=50, gamma=0.9)
    scheduler_d = torch.optim.lr_scheduler.StepLR(optimizer_d, step_size=50, gamma=0.9)    

    loss_d_list, loss_ge_list, anomaly_score_list = [], [], []


    for epoch in range(int(EPOCHS)):
        loss_d_sum = 0
        loss_ge_sum = 0
        anomaly_score_sum = 0

        for i,(x, x_val) in enumerate(zip(train_loader, val_loader)):
            
            model_G.train()
            model_D.train()
            model_E.train()
            # set values
            y_true = Variable(torch.ones(x.size()[0])).to(device)
            y_fake = Variable(torch.zeros(x.size()[0])).to(device)
            
            x = Variable(x).to(device)
            z = Variable(init.normal(torch.Tensor(x.size()[0],EMBED_SIZE, 1, 1),mean=0,std=0.1)).to(device)
            
            # noise for discriminator
            noise1 = Variable(torch.Tensor(x.size()).normal_(0, 0.1 * (EPOCHS - epoch) / EPOCHS),
                            requires_grad=False).to(device)
            noise2 = Variable(torch.Tensor(x.size()).normal_(0, 0.1 * (EPOCHS - epoch) / EPOCHS),
                            requires_grad=False).to(device)

            # discriminator
            optimizer_d.zero_grad()
            
            E_x = model_E(x) 
            p_true, _ = model_D(x + noise1, E_x)
            
            G_z = model_G(z)
            p_fake, _ = model_D(G_z + noise2, z)
            
            loss_d = criterion(p_true, y_true) + criterion(p_fake, y_fake)
            loss_d.backward(retain_graph=True)
            optimizer_d.step()
            
            # generator and encoder
            optimizer_ge.zero_grad()
            
            G_E_x = model_G(E_x)
            E_G_z = model_E(G_z)
        
            p_true, _ = model_D(x + noise1, E_x)
            
            # G_z = model_G(z)
            p_fake, _ = model_D(G_z + noise2, z)
            
            
            loss_ge_1 = criterion(p_fake, y_true) + criterion(p_true, y_fake)
            loss_ge_2 = criterion_L1(x, G_E_x) +  criterion_L1(z, E_G_z)
            
            alpha = 0.01
            
            loss_ge = (1 - alpha)*loss_ge_1 + alpha*loss_ge_2
            loss_ge.backward(retain_graph=True)
            optimizer_ge.step()
            
            
            loss_d_sum += loss_d.item()
            loss_ge_sum += loss_ge.item()
            
            # record anomaly score
            
            model_G.eval()
            model_D.eval()
            model_E.eval()
            x_val = Variable(x_val).to(device)
            E_x_val = model_E(x_val)
            G_E_x_val = model_G(E_x_val)
            anomaly_score_sum += Anomaly_score(x_val, E_x_val, G_E_x_val)

                
            # save images
            if i == 0:
                
                model_G.eval()
                model_D.eval()
                model_E.eval()
            
                save_image_size_for_z = min(BATCH_SIZE, 8)
                save_images = model_G(z)
                save_image(save_images[:save_image_size_for_z], f"{OUT_IMAGE_Z_DIR}/epoch_{epoch}.png", nrow=4)

                save_image_size_for_recon = min(BATCH_SIZE, 8)
                images = x[:save_image_size_for_recon]
                G_E_x = model_G(model_E(images))
                diff_images = torch.abs(images - G_E_x)
                comparison = torch.cat([images , G_E_x, diff_images]).to("cpu")
                save_image(comparison, f"{OUT_IMAGE_RECONSTRUCT}/epoch_{epoch}.png", nrow=save_image_size_for_recon)

        scheduler_ge.step()
        scheduler_d.step()
            
        # record loss
        loss_d_mean = loss_d_sum / len(train_loader)
        loss_ge_mean = loss_ge_sum / len(train_loader)
        anomaly_score_mean = anomaly_score_sum / len(train_loader)
        
        print(f"{epoch}/{EPOCHS} epoch ge_loss: {loss_ge_mean:.3f} d_loss: {loss_d_mean:.3f} anomaly_score: {anomaly_score_mean:.3f}")
        
        loss_d_list.append(loss_d_mean)
        loss_ge_list.append(loss_ge_mean)
        anomaly_score_list.append(anomaly_score_mean)
        
        # save model
        if (epoch + 1) % 10 == 0:
            torch.save(model_G.state_dict(),f'{OUT_MODEL_DIR}/Generator_{epoch + 1}.pkl')
            torch.save(model_E.state_dict(),f'{OUT_MODEL_DIR}/Encoder_{epoch + 1}.pkl')
            torch.save(model_D.state_dict(),f'{OUT_MODEL_DIR}/Discriminator_{epoch + 1}.pkl')


    # ロスのグラフを出力
    fig = plt.figure(figsize=(12,8*2))
    # ロスプロット
    ax1 = fig.add_subplot(2,1,1)
    ax1.set_title(f"epoch{int(EPOCHS)}:Loss chart")
    ax1.plot(range(len(loss_ge_list)),loss_ge_list,label="Loss_ge",color="blue")
    ax1.plot(range(len(loss_d_list)),loss_d_list,label="Loss_d",color="red")
    ax1.legend()
    # 異常値プロット
    ax2 = fig.add_subplot(2,1,2)
    ax2.set_title(f"epoch{int(EPOCHS)}:Anomaly score")
    ax2.plot(range(len(anomaly_score_list)),anomaly_score_list,label="anomaly score",color="green")
    ax2.legend()

    fig.savefig(os.path.join(OUT_MODEL_DIR,f"Loss chart_e.jpg"))
    # メモリ対策
    plt.close()
    # 実行時間表示
    print("全体実行時間:",time.time()-st)