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

# 異常値の関数
def Anomaly_score(x, E_x, G_E_x, Lambda=0.1):
    _,x_feature = model_D(x, E_x)
    _,G_E_x_feature = model_D(G_E_x, E_x)
    residual_loss = criterion_L1(x, G_E_x)
    discrimination_loss = criterion_L1(x_feature, G_E_x_feature)
    
    total_loss = (1-Lambda)*residual_loss + Lambda*discrimination_loss
    total_loss = total_loss.item()
    
    return total_loss
    
parser = argparse.ArgumentParser(description='efficientgan test script')
parser.add_argument('--config_file', type=str, default="efgan_config.yml")
args = parser.parse_args()

if __name__ == '__main__':
    # 実行時間計測開始
    st = time.time()
    # 作業ディレクトリ
    work_dir="."
    # cudaの設定
    # GPUが使えるならGPUで、そうでないならCPUで計算する    
    device = "cuda" if torch.cuda.is_available() else "cpu"  
    ## コンフィグファイル読み込み
    config_path = os.path.join(os.path.join(work_dir,"config"),args.config_file)
    config_dict = yaml.safe_load(open(config_path, 'r'))

    ## 辞書から変数読み込み
    IMAGE_SIZE = config_dict["image_size"]
    EMBED_SIZE = config_dict["embed_size"]
    BATCH_SIZE = config_dict["batch_size"]
    LR = config_dict["lr"]
    EPOCHS = config_dict["epochs"]
    OUT_MODEL_DIR = config_dict["out_model_dir"]
    OUT_TEST_RESULTS = config_dict["out_test_dir"]
    TEST_IMAGE_NUMBER = config_dict["test_image_number"]
    LOAD_EPOCH = config_dict["load_epoch"]
    ## model読み込みフォルダパス
    OUT_MODEL_DIR = os.path.join(work_dir,OUT_MODEL_DIR)
    ## 出力フォルダパス
    OUT_TEST_RESULTS = os.path.join(work_dir,OUT_TEST_RESULTS)
    os.makedirs(OUT_TEST_RESULTS,exist_ok=True)

    # transform_dictを再度設定
    transform_dict = {
        "train": transforms.Compose(
            [
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),  # IMAGE_SIZEにreshape
                transforms.RandomHorizontalFlip(), # ランダムに左右反転を行う
                transforms.ToTensor(),
            ]
        ),
        "test": transforms.Compose(
            [
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),  # IMAGE_SIZEにreshape
                transforms.ToTensor(),
            ]
        ),
    }
    

    # 学習したモデルの読み込み

    model_G = Generator(EMBED_SIZE=EMBED_SIZE).to(device)
    model_G.load_state_dict(torch.load(f"{OUT_MODEL_DIR}/Generator_{LOAD_EPOCH}.pkl"))
    model_G.eval()


    model_E = Encoder(EMBED_SIZE=EMBED_SIZE).to(device)
    model_E.load_state_dict(torch.load(f"{OUT_MODEL_DIR}/Encoder_{LOAD_EPOCH}.pkl"))
    model_E.eval()

    model_D = Discriminator(EMBED_SIZE=EMBED_SIZE).to(device)
    model_D.load_state_dict(torch.load(f"{OUT_MODEL_DIR}/Discriminator_{LOAD_EPOCH}.pkl"))
    model_D.eval()
    print("load model")

    ## 損失関数と最適化手法
    criterion = nn.BCELoss()  # 評価関数=二値交差検証
    criterion_L1 = nn.L1Loss(reduction="sum") # 異常スコア計測用

    ## 様々な画像で実行
    # 後日"."に変更(pyファイル用)
    work_dir = "."
    random_image_size = TEST_IMAGE_NUMBER

    test_root_normal = os.path.join(work_dir,'test')
    test_dataset_normal = LoadFromFolder(test_root_normal, transform=transform_dict["test"])
#    print("test_dataset_normal:\n",test_dataset_normal)

    test_images_normal = random.sample(list(test_dataset_normal), random_image_size)
#    print("list(test_dataset_normal):\n",list(test_dataset_normal))
#    print("test_images_normal:\n",test_images_normal)

    # うまく再現され、異常スコアが低くなっていれば成功
    for idx in range(len(test_images_normal)):

        x = test_images_normal[idx].view(1, 3, IMAGE_SIZE, IMAGE_SIZE).to(device)
#        print("test_images_normal[idx]:",test_images_normal[idx])
        E_x = model_E(x)
        G_E_x = model_G(E_x)
        loss = Anomaly_score(x, E_x, G_E_x)
        diff_img = torch.abs(x - G_E_x)

        print(f"Anomary_score = {loss:.3f}")

        fig,axes = plt.subplots(1,3,figsize=(18,6))
        axes[0].set_title("original image")
        axes[0].imshow(((x.to("cpu").detach().numpy()).squeeze().transpose(1,2,0)*255).astype(np.uint8))
        axes[0].axis("off")
        axes[1].set_title("generated image")
        axes[1].imshow(((G_E_x.to("cpu").detach().numpy()).squeeze().transpose(1,2,0)*255).astype(np.uint8))
        axes[1].axis("off")
        axes[2].set_title("diff image")
        axes[2].imshow(((diff_img.to("cpu").detach().numpy()).squeeze().transpose(1,2,0)*255).astype(np.uint8))
        axes[2].axis("off")

    #    plt.figure(figsize=(12, 4))
        plt.suptitle(f"Anomary_score = {loss:.3f}")
    #    plt.imshow((joined_image * 255).astype(np.uint8))
        plt.savefig(os.path.join(OUT_TEST_RESULTS,f"comparison_image_{idx}.jpg"))
#        plt.show()
        plt.close()
    # 全体実行時間表示
    print("全体実行時間:",time.time()-st)