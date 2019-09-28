import mxnet as mx


class MXDataset(mx.gluon.data.Dataset):
    def __init__(self, root, df, labels_ids, image_transform):
        super().__init__()
        self._root = root
        self._df = df[:15]
        self._labels_ids = labels_ids
        self._image_transform = image_transform

    def __len__(self):
        return len(self._df)

    def __getitem__(self, idx: int):
        item = self._df.iloc[idx]
        
        image = mx.image.imread(self._root.joinpath(f'{item.id}'))
        image = self._image_transform(image)
        
        return image, mx.nd.array(item[self._labels_ids].values)
    
    
class MXDatasetTest(mx.gluon.data.Dataset):
    def __init__(self,
                 root,
                 df,
                 image_transform,
                 extension: str = 'png'
                ):
        super().__init__()
        self._root = root
        self._df = df
        self._image_transform = image_transform
        self._extension = extension

    def __len__(self):
        return len(self._df)

    def __getitem__(self, idx: int):
        item = self._df.iloc[idx]
        
        image = mx.image.imread(self._root.joinpath(f'{item.id}.{self._extension}'))
        image = self._image_transform(image)
        
        return image