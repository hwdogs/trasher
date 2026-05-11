import torch
import netron
from model.model import MiniMobileResNet

model = MiniMobileResNet()

input = torch.ones((1,3,224,224))

torch.onnx.export(model, input, f='MiniMobileResNet.onnx')
netron.start('MiniMobileResNet.onnx')