from core.modelnet_data import DataModule

if __name__=="__main__":
    dm = DataModule(data_dir="data/modelnet40", batch_size=8)

    print(dm.classes)

    dm.setup(stage=None)

    dl = dm.val_dataloader()

    (points, features), label = next(iter(dl))

    print(points.shape)
    print(features.shape)
    print(label)
