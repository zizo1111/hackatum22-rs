import torch
from torch import nn

from resnet import resnet50
from dataset import MicrowaveDataset


class PoseEstimator(nn.Module):
    def __int__(self):
        super().__init__()
        self.backbone = resnet50()
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 6)


if __name__ == '__main__':
    # dummy
    #dummy = torch.rand([1, 6, 129, 129])
    # pose_estimator = PoseEstimator()
    dataset = MicrowaveDataset('../data/dummy_measurements/volumes',
                               '../data/dummy_measurements/labels_transformed.json')

    model = resnet50(num_classes=6)
    for i in range(len(dataset)):
        ret = dataset.__getitem__(i)
        output = model.forward(ret["inputs"])
        print(ret['label'], ret['inputs'].shape)
        print("output shape: ", output.shape)

