from mxnet.gluon.data.vision import transforms


def get_train_transform(resize, crop, scale, mean, std):
    return transforms.Compose([
        #  transforms.Resize(resize),
        #  transforms.CenterCrop(crop),
        transforms.RandomResizedCrop(crop, scale=scale),  # no ratio
        transforms.RandomFlipLeftRight(),
        #  transforms.RandomColorJitter(brightness=lighting_param,
        #                               contrast=lighting_param,
        #                               saturation=lighting_param),
        #  transforms.RandomLighting(lighting_param),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
])


def get_test_transform(resize, crop, mean, std):
    return transforms.Compose([
        #  transforms.Resize(resize),
        transforms.CenterCrop(crop),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])