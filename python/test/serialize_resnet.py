import torchvision
import sys
sys.path.append(".")
import icdl
import icdl.pytorch_serialize_utils as s_utils

if __name__ == "__main__":
    model = torchvision.models.resnet18()
    s_utils.serialize_pytorch_float_model(model, "res18_float.icdl_model")