## 生成器
from torch import nn, optim

#Generatorクラスではnn.Moduleを親クラスとしてinitを設定している。
#__init__では、使用するレイヤーのインスタンスを生成し、forward()で決めた順番に適用していく。
#super().__init__()でオーバーライドは必須

class Generator(nn.Module):
    def __init__(self,EMBED_SIZE):
        super().__init__()
# 一連のネットワークの中にレイヤーを定義している。転置畳み込みを用いて6x6のzから～96x96の画像を生成している。
        self.main = nn.Sequential(
# Conv2dによる出力サイズと整合性をとるため、dilation[=基本1] * (kernel_size - 1) - paddingが実際のパディング幅となる。
# つまり、このpaddingは混乱を避けるために『畳み込みを行う際のパディング』を指定しています。
# EMBED_SIZE=画像の入力チャンネル数、256=出力時のチャンネル数
# チャンネル数＝フィルターの数、転置畳み込みを行うフィルターの数を表すが、rgb画像の場合、最終的にチャンネル数=3となる(Rに1フィルター、Gに1フィルター、Bに1フィルター)はずなので
            nn.ConvTranspose2d(EMBED_SIZE, 2048, kernel_size=3, stride=1, padding=0, bias=False),  # 6x6 => 3x3
            nn.BatchNorm2d(2048),
# 勾配がスパースになるReLUよりも0.1 x αを実現するleakyReluが有効
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(2048, 1024, kernel_size=5, stride=2, padding=1, bias=False),  # 12x12 => 7x7
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, bias=False),  # 24x24 => 13x13
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, bias=False),  # 48x48 => 25x25
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, bias=False),  # 96x96 => 49x49
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, bias=False),  # None => 97x97
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2, padding=1, bias=False),  # None => 193x193
#            nn.BatchNorm2d(32),
#            nn.LeakyReLU(0.2),
#            nn.ConvTranspose2d(32, 3, kernel_size=60, stride=1, padding=1, bias=False),  # None => 250x250
            # Gの出力が[-1,1]となるように,outputをTanhにする。これがDのインプットとなる。           
            nn.Tanh(),
        )

    def forward(self, z):
# 上記のネットワークにノイズzを流す
        out = self.main(z)
# ネットワークに流した結果を返す。
        return out