## エンコーダー
from torch import nn, optim

class Encoder(nn.Module):
    def __init__(self,EMBED_SIZE):
        super().__init__()
        self.main = nn.Sequential(
# 画像からzに変換
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False), # 48x48 => 97 x 97 (input:193x193)
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),  
            
            nn.Conv2d(64, 128 , kernel_size=3, stride=2, padding=1, bias=False), # 24x24 => 49 x 49
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),  
            
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False), # 12x12 => 25 x 25
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),  
            
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False), # 6x6 => 13 x 13
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),  
            
            nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1, bias=False), # 1x1 => 7 x 7
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2),  
            
            nn.Conv2d(1024, 2048, kernel_size=5, stride=2, padding=1, bias=False), # 1x1 => 3 x 3
            nn.BatchNorm2d(2048),
            nn.LeakyReLU(0.2),  

            nn.Conv2d(2048, 4096, kernel_size=3, stride=1, padding=0, bias=False), # 1x1 => 1 x 1
            nn.BatchNorm2d(4096),
            nn.LeakyReLU(0.2),              
        )

        # 潜在変数ｚを生成
        # 最後にEMBED_SIZEにチャンネル数を設定している。元のサイズはどうなるか？
        self.last = nn.Sequential(nn.Conv2d(512, EMBED_SIZE, kernel_size=1, stride=1, bias=False))

    def forward(self, x):

        out = self.main(x)
        out = self.last(out)
        out = out.view(out.size()[0], -1, 1, 1)
        return out        