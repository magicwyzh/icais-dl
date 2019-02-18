import torchvision
import sys
sys.path.append(".")
import icdl
import icdl.pytorch_serialize_utils as s_utils
from icdl.pb_utils import Serializer
from icdl.pb_utils import FP32Descript
import torch
from collections import OrderedDict
from functools import partial
def serialize_named_tensors(tensor_map, file_name):
    assert isinstance(tensor_map, OrderedDict)
    worker = Serializer()
    params_map = OrderedDict()
    for name, tensor in tensor_map.items():
        size = list(tensor.size())
        if len(size) == 0:
            size = [1]
        params_map[name] = [size, FP32Descript(), list(tensor.storage()), "DENSE_LAYOUT"]
        print("Setting data of: " + name)
    print("Writting to file: " + file_name + "...")
    worker.serialize_compute_graph_to_file(file_name, params_map)
    print("Done!")

if __name__ == "__main__":
    torch.manual_seed(22)
    images = torch.Tensor(1,3,224,224).normal_()
    model = torchvision.models.resnet18()
    model = model.eval()
    s_utils.serialize_pytorch_float_model(model, "../test/test_data/res18_float.icdl_model")
    s_utils.serialize_pytorch_tensor(images, "../test/test_data/images.icdl_tensor")
    layer_outputs = OrderedDict()
    def hook(module, inputs, outputs, module_name):
        assert isinstance(outputs, torch.Tensor)
        layer_outputs[module_name] = outputs.clone()
        print("module_name:" + module_name + ", size: ", outputs.size())
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:
            # leaf module
            name = name.replace(".", "->")
            partial_hook = partial(hook, module_name = name)
            module.register_forward_hook(partial_hook)

    output = model(images)
    print("Serializing layer outputs:")
    serialize_named_tensors(layer_outputs, "../test/test_data/res18_layer_outs.icdl_model")
    s_utils.serialize_pytorch_tensor(output, "../test/test_data/output.icdl_tensor")
    