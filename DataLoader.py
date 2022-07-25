## データ読み込み->DataLoaderへ渡すクラス(trainデータは水平反転)
from natsort import natsorted
from torch.utils.data import Dataset
import os
from PIL import Image

class LoadFromFolder(Dataset):
    def __init__(self, main_dir, transform):
        self.main_dir = main_dir
        self.transform = transform
        # testディレクトリ内のファイル(jpg)一覧リストを作成して、
        # natsortedで数値順に並べ替えている。(通常のsort()はアルファベット順で並べる)
        all_imgs = natsorted(os.listdir(main_dir))
        self.all_imgs_name = natsorted(all_imgs) # 数字ソートしたファイル名リストをさらに数字順でソートしている？
#        print("self.all_imgs_name:\n",self.all_imgs_name)
                # 各画像へのパスをリスト化している
        self.imgs_loc = [os.path.join(self.main_dir, i) for i in self.all_imgs_name]
#        print("self.imgs_loc:\n",self.imgs_loc)

    def __len__(self):
        return len(self.all_imgs_name)
    
    def load_image(self, path):
        image = Image.open(path).convert("RGB")
        tensor_image = self.transform(image)
        return tensor_image
    
    def __getitem__(self, idx):
        
#        print("__getitem__called!\n")
        # 後ほどsliceで画像を複数枚取得したいのでsliceでも取れるようにする
        # インデックスの書き方がsliceクラスの場合(例：test_dataset_normal[1,10,1])
        # sliceの書き方に従い複数のパスをpathsに入れる。
        # その後、pathsの中から1つずつパスを取り出し、画像を読み込んで、リストに入れていく。
        # リスト内のテンソルを結合して、形状を変換している。
        if type(idx) == slice:
            paths = self.imgs_loc[idx]
            tensor_image = [self.load_image(path) for path in paths]
            tensor_image = torch.cat(tensor_image).reshape(len(tensor_image), *tensor_image[0].shape)
        # インデックスの書き方が整数の場合(例：test_dataset_normal[0])
        # その番号の画像パスから画像を1枚読み込んでいる。
        elif type(idx) == int:
#            print("designated as int!\n")
            # 画像パスのリストから[idx]番目をpathとして、画像を読み込んでいる
            path = self.imgs_loc[idx]
            tensor_image = self.load_image(path)
        return tensor_image
