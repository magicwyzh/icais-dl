import torch
import sys
sys.path.append(".")
import icdl
import icdl.pytorch_serialize_utils as s_utils
from icdl.pb_utils import Serializer
from icdl.pb_utils import FixpointRepresent
import icdl.pb_utils as pb_utils

def serialize_pytorch_fixp_tensor(tensor, descriptor, file_name):
    assert isinstance(tensor, torch.Tensor)
    assert isinstance(descriptor, pb_utils.TensorDataDescriptor)
    worker = Serializer()
    tensor_msg = worker.serialize_tensor(list(tensor.size()), descriptor, list(tensor.storage()),"DENSE_LAYOUT")
    #print("Writting to file...")
    with open(file_name, "wb") as f:
        s = tensor_msg.SerializeToString()
        f.write(s)

if __name__ == "__main__":
    torch.manual_seed(22)
    tensor = torch.Tensor(1,3,2,4).normal_(0, 100).char()
    represent = FixpointRepresent([8], [True], [0], [0.1375], [0])
    descriptor = pb_utils.TensorDataDescriptor("FIXPOINT", represent)
    serialize_pytorch_fixp_tensor(tensor, descriptor, "../test/test_data/fixp_tensor.icdl_tensor")
