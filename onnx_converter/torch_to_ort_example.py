import sys
import os
import cv2
import argparse
import onnx
import torch
from onnx import helper
from onnx import TensorProto
from onnx import numpy_helper
import numpy as np
import backbones

print('onnx version:', onnx.__version__)

assert onnx.__version__ >= '1.2.1', 'onnx version should >= 1.2.1'


parser = argparse.ArgumentParser(description='convert arcface_torch models to onnx')
# general
parser.add_argument('network', default='iresnet100', help='network definition')
parser.add_argument('weight', default='modelzoo/torch_model/ms1mv3_arcface_r100_fp16/backbone.pth', help='torch weights to load.')
parser.add_argument('output', default='./r100a.onnx', help='path to write onnx model.')
parser.add_argument('--opset', type=int, default=11, help='opset version.')
#parser.add_argument('--input-image', type=str, default='../../deploy/Tom_Hanks_54745.png', help='example input image')
parser.add_argument('--input-image', type=str, default='', help='example input image')
args = parser.parse_args()
#input_shape = (1,) + tuple( [int(x) for x in args.input_shape.split(',')] )
#print('input-shape:', input_shape)
weight = torch.load(args.weight)
net = eval("backbones.{}".format(args.network))(False)
net.load_state_dict(weight)
#net = torch.nn.DataParallel(net)
net.eval()

if len(args.input_image)>0:
    img = cv2.imread(args.input_image)
else:
    img = np.random.randint(0, 255, size=(112,112,3), dtype=np.uint8)
img = img[:,:,::-1].astype(np.float32)
img = (img/255.0 - 0.5) / 0.5 #torch style norm
img = img.transpose( (2,0,1) )
#one_img = torch.from_numpy(img).unsqueeze(0).float().requires_grad_(True)
one_img = torch.from_numpy(img).unsqueeze(0).float()


torch.onnx.export(
        net,
        one_img,
        args.output,
        keep_initializers_as_inputs=False,
        verbose=True,
        opset_version=args.opset)

model = onnx.load(args.output)
graph = model.graph

#support batch-inference
graph.input[0].type.tensor_type.shape.dim[0].dim_param = 'None'
onnx.save(model, args.output)

