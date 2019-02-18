from .pb_utils import Serializer
from .pb_utils import FP32Descript
import torch
import torch.nn as nn
from collections import OrderedDict
def pytorch_tensor_name_to_icdl_tensor_name(name):
    '''
        @param name: name string from model.state_dict().keys()
    '''
    parts = name.split(".")
    s = ""
    for i in range(len(parts) - 1):
        s += parts[i]
        s += "->"
    s = s[:-2] + "." + parts[-1]
    return s

def serialize_pytorch_float_model(model, file_name):
    assert isinstance(model, nn.Module)
    params_map = OrderedDict()
    worker = Serializer()
    print("Start serializing Pytorch float model...")
    for name, state in model.state_dict().items():
        name = pytorch_tensor_name_to_icdl_tensor_name(name)
        size = list(state.size())
        if len(size) == 0:
            size = [1]
        params_map[name] = [size, FP32Descript(), list(state.storage()), "DENSE_LAYOUT"]
    print("Writting to file...")
    worker.serialize_compute_graph_to_file(file_name, params_map)
    print("Done!")

def serialize_pytorch_tensor(tensor, file_name):
    assert isinstance(tensor, torch.Tensor)
    worker = Serializer()
    print("Start serializing to " + file_name + "...")
    tensor_msg = worker.serialize_tensor(list(tensor.size()), FP32Descript(), list(tensor.storage()),"DENSE_LAYOUT")
    print("Writting to file...")
    with open(file_name, "wb") as f:
        s = tensor_msg.SerializeToString()
        f.write(s)
    print("Done!")
