import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import cv2


class ResNetUSCL(nn.Module):
    ''' The ResNet feature extractor + projection head + classifier for USCL '''

    def __init__(self, base_model, out_dim, pretrained_path=None):
        super(ResNetUSCL, self).__init__()

        self.resnet_dict = {"resnet18": models.resnet18(pretrained=False),
                            "resnet50": models.resnet50(pretrained=False)}
        resnet = self._get_basemodel(base_model)
        num_ftrs = resnet.fc.in_features

        self.features = nn.Sequential(*list(resnet.children())[:-1])  # discard the last fc layer

        # projection MLP
        self.linear = nn.Linear(num_ftrs, out_dim)

        # classifier
        num_classes = 12
        self.fc = nn.Linear(out_dim, num_classes)
        if pretrained_path is not None:
            _state_dict = torch.load(pretrained_path)
            _new_dict = {k: _state_dict[k] for k in list(_state_dict.keys())
                         if not (k.startswith('l')
                                 | k.startswith('fc'))}  # discard MLP and fc
            _model_dict = self.state_dict()
            _model_dict.update(_new_dict)
            self.load_state_dict(_model_dict)
            print('\nModel parameters loaded.\n')
        else:
            print('\nRandom initialize model parameters.\n')



    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
            print("Feature extractor:", model_name)
            return model
        except:
            raise ("Invalid model name. Check the config file and pass one of: resnet18 or resnet50")

    def forward(self, x):
        h = self.features(x)
        h = h.view(h.size(0), -1)
        # x = self.linear(h)
        return h


if __name__ == '__main__':
    net = ResNetUSCL(base_model='resnet18', out_dim=256, pretrained=False)
    state_dict = torch.load('./best_model.pth')
    new_dict = {k: state_dict[k] for k in list(state_dict.keys())
                if not (k.startswith('l')
                        | k.startswith('fc'))}  # # discard MLP and fc
    model_dict = net.state_dict()

    model_dict.update(new_dict)
    net.load_state_dict(model_dict)
    net.eval()
    valid_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25])
    ])
    # 载入并处理图片
    image = Image.open('./img1.bmp')
    image = valid_transform(image)
    image = image.unsqueeze(0)
    print(image.size())

    # 使用模型进行预测
    with torch.no_grad():
        outputs = net(image)
        # cv2.imwrite('./result.png', outputs.data)
        # print(outputs)
