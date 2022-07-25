## 識別器
import torch
from torch import nn, optim
from torch.nn import functional as F

class Discriminator(nn.Module):
    def __init__(self,EMBED_SIZE):
        super().__init__()
        
        # x層は5層のCNN(サイズを小さく1x1までにしている,出力形状[1,256,1,1])
        self.x_layer = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),  # 48x48 => 97 x 97 (input:193x193)
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(p=0.3),
            # 元Dropout
#            nn.Dropout2d(p=0.3),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 24x24 => 49 x 49
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(p=0.3),
            # 元Dropout
#            nn.Dropout2d(p=0.3),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # 12x12 => 25 x 25
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(p=0.3),
            # 元Dropout
#            nn.Dropout2d(p=0.3),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),  # 6x6 => 13 x 13
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(p=0.3),
# 新規層(size=193用)
            nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1),  # none => 7 x 7
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(p=0.3),
            # 元Dropout
#            nn.Dropout2d(p=0.3),
# 新規層2(size=193用)
            nn.Conv2d(1024, 2048, kernel_size=5, stride=2, padding=1),  # none => 3 x 3
            nn.BatchNorm2d(2048),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(p=0.3),
            # 元Dropout
#            nn.Dropout2d(p=0.3),
            nn.Conv2d(2048, 2048, kernel_size=3, stride=1),  # 1x1
        )
        # z層は1層(出力形状,[1,256,1,1])
        # EMBED_SIZEはフィルターの数なので、増やせば表現力が増すはず。ただ、一層のみフィルターが多いことでどれだけ影響があるのかはわからない
        self.z_layer = nn.Sequential(
            nn.Conv2d(EMBED_SIZE, 2048, kernel_size=1, stride=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(p=0.2),
            # 元Dropout
#            nn.Dropout2d(p=0.2),
        )
        self.last1 = nn.Sequential(
            nn.Conv2d(4096, 4096, kernel_size=1, stride=1),
            nn.LeakyReLU(0.1, inplace=False),
            nn.Dropout2d(p=0.2),
            # 元Dropout
#            nn.Dropout2d(p=0.2),
        )
        # この層で4096チャンネルを1チャンネルにしている。4096の値をどう計算して1つにまとめている？平均？
        self.last2 = nn.Sequential(
            nn.Conv2d(4096, 1, kernel_size=1, stride=1),
        )

    def forward(self, x, z):
        
        # オリジナル画像(x)や+noize、生成画像などをx層の入力として、1x1を出力
        output_x = self.x_layer(x)
        print("output_x size:\n",output_x.size())
        # ノイズzやエンコードデータなどをz層の入力として、そのまま1x1を出力
        output_z = self.z_layer(z)
        print("output_z size:\n",output_z.size())
        
        # x出力とz出力をフィルター数の方向に結合する。(1365 + 1365 = 2730)
        # フィルター数が2730なので、2730パターンの数字が入っている
        concat_x_z = torch.cat((output_x, output_z), 1)
        print("concat_x_z size:\n",concat_x_z.size())
        # last1を通すとデフォルトでは[1,2730,1,1]となる。
        output = self.last1(concat_x_z)
        print("size after self.last1:\n",output.size())
        
        # 最終層(last2)の1層前に特徴として取り出す。サイズは(バッチ数の次元(画像はデータローダーでバッチサイズ=8として8枚->1枚に結合されているので、すでに1。8ではない) x チャンネルの次元(2730))[1,2730]となる。
        feature = output.view(output.size()[0], -1)
        print("feature.size():\n",feature.size())

        # last2を通すとデフォルトでは[1,1,1,1](バッチ,フィルタ数,幅,高さ)となる。
        output = self.last2(output)
        print("size after self.last2:\n",output.size())
        print("before sigmoid:\n",output)
        # シグモイドを通した後も形状は[1,1,1,1]のまま
        output = F.sigmoid(output)
        print("size after last sigmoid:\n",output.size())
        print("after sigmoid:\n",output)
        print("after squeeze:\n",output.squeeze())
        print("size after squeeze:\n",output.squeeze().size())
        
        ## ①確率、②特徴を返す。
        # サイズが1ならsqueeze()を通すことで1次元に戻すことができる。
        return output.squeeze(), feature