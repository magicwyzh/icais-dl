import sys
import os
csfp = os.path.abspath(os.path.dirname(__file__))
if csfp not in sys.path:
    sys.path.insert(0, csfp)
import ComputeGraph_pb2 as cg_pb
import Tensor_pb2 as tensor_pb
import struct
class DataRepresent(object):
    def __init__(self):
        super().__init__()

class FloatpointRepresent(DataRepresent):

    def __init__(self, total_bits = 32, is_signed = True, exp_bits = 8, mantissa_bits=23):
        super().__init__()
        self.total_bits = total_bits
        self.is_signed = is_signed
        self.exp_bits = exp_bits
        self.mantissa_bits = mantissa_bits

class FixpointRepresent(DataRepresent):
    '''
        @param total_bits& is_signed& frac_point_locs & scalars& zero_points should 
        all be sequence like list.
    '''
    def __init__(self, total_bits, is_signed, frac_point_locs, scalars, zero_points):
        super().__init__()
        assert isinstance(total_bits, list)
        assert isinstance(is_signed, list)
        assert isinstance(frac_point_locs, list)
        assert isinstance(scalars, list)
        assert isinstance(zero_points, list)
        self.total_bits = total_bits
        self.is_signed = is_signed
        self.frac_point_locs = frac_point_locs
        self.scalars = scalars
        self.zero_points = zero_points


class TensorDataDescriptor(object):
    '''
    @param dtype: string {"FLOAT_32", "FLOAT_16", "FIXPOINT"}
    '''
    def __init__(self, dtype, data_represent):
        super().__init__()
        assert dtype in {"FLOAT_32", "FLOAT_16", "FIXPOINT"}
        self.dtype = dtype
        if dtype in {"FLOAT_32", "FLOAT_16"}:
            assert isinstance(data_represent, FloatpointRepresent)
            self.data_represent = data_represent
        else: 
            assert isinstance(data_represent, FixpointRepresent)
            self.data_represent = data_represent

# for brevity
def FP32Descript():
    return TensorDataDescriptor("FLOAT_32", FloatpointRepresent())

class Serializer(object):
    
    def __init__(self):
        super().__init__()
    
    def _float_list_to_bytes(self, float_list, half_prec = False):
        #import ipdb;ipdb.set_trace() 
        assert isinstance(float_list, list)
        
        num_data = len(float_list)
        if(not half_prec):
            fmt = "<" + str(num_data) + "f"
        else:
            fmt = "<" + str(num_data) + "e"
        data_bytes = struct.pack(fmt, *float_list)
        return data_bytes
    
    def _integer_list_to_bytes(self, integer_list, descriptor):
        assert isinstance(integer_list, list) and isinstance(float_list[0], int)
        c = ""
        if descriptor.data_represent.total_bits<=8 and descriptor.data_represent.is_signed:
            c = "b"#signed char
        elif descriptor.data_represent.total_bits<=8 and not descriptor.data_represent.is_signed:
            c = "B" #unsigned char
        elif descriptor.data_represent.total_bits<=16 and descriptor.data_represent.is_signed:
            c = "h" #signed short
        elif descriptor.data_represent.total_bits<=16 and not descriptor.data_represent.is_signed:
            c = "H" #unsigned short
        else:
            raise ValueError("Invalid data descriptor for integer_list_to_byte")
        num_data = len(integer_list)
        fmt = "<" + str(num_data) + c
        data_bytes = struct.pack(fmt, *integer_list)
        return data_bytes

    def serialize_tensor_storage(self, data_descriptor, storage_data):
        '''
        @param descriptor : instance of TensorDataDescriptor
        @param storage_data : list of floats...
        return : protobuf object for TensorStorage
        '''
        descriptor = data_descriptor
        assert isinstance(descriptor, TensorDataDescriptor) 
        assert isinstance(storage_data, list)
        
        storage_pb = tensor_pb.TensorStorage()
        storage_pb.data_descriptor.dtype = tensor_pb.TensorDataDescriptor.TensorDataType.Value(descriptor.dtype)
        if descriptor.dtype in {"FLOAT_32", "FLOAT_16"}:
            storage_pb.data_descriptor.flo_point.total_bits = descriptor.data_represent.total_bits
            storage_pb.data_descriptor.flo_point.is_signed = descriptor.data_represent.is_signed
            storage_pb.data_descriptor.flo_point.exp_bits = descriptor.data_represent.exp_bits
            storage_pb.data_descriptor.flo_point.mantissa_bits = descriptor.data_represent.mantissa_bits
            storage_pb.data = self._float_list_to_bytes(storage_data, descriptor.dtype == "FLOAT_16")
        else:
            storage_pb.data_descriptor.fix_point.total_bits = descriptor.data_represent.total_bits
            storage_pb.data_descriptor.fix_point.is_signed = descriptor.data_represent.is_signed
            storage_pb.data_descriptor.fix_point.frac_point_locs = descriptor.data_represent.frac_point_locs
            storage_pb.data_descriptor.fix_point.scalars = descriptor.data_represent.scalars
            storage_pb.data_descriptor.fix_point.zero_points = descriptor.data_represent.zero_points
            storage_pb.data = self._integer_list_to_bytes(storage_data, descriptor)
        return storage_pb

    def serialize_tensor(self, tensor_size, data_descriptor, storage_data, mem_layout="DENSE_LAYOUT"):
        '''
        @param mem_layout: one of  "DENSE_LAYOUT", "SPARSE_LAYOUT", "INVALID_LAYOUT"
        @param tensor_size: sequence of int
        '''
        # not use 'tensor_pb' here due to name conflict with the python module.
        tensor_message = tensor_pb.Tensor()
        #tensor_message.tensor_size = tensor_size
        for dim in tensor_size:
            tensor_message.tensor_size.append(dim)
        tensor_message.storage.CopyFrom(self.serialize_tensor_storage(data_descriptor, storage_data))
        tensor_message.mem_layout = tensor_pb.Tensor.TensorMemLayout.Value(mem_layout)
        
        return tensor_message
    
    def serialize_compute_graph(self, name_data_map):
        assert isinstance(name_data_map, dict)
        graph_param_pb = cg_pb.GraphParams()
        for param_name, data in name_data_map.items():
            size, data_descriptor, storage_data, mem_layout = data
            #print("Serializing: " + param_name + " ...")
            serialized_tensor = self.serialize_tensor(size, data_descriptor, storage_data, mem_layout)
            graph_param_pb.graph_params[param_name].CopyFrom(serialized_tensor)
        
        return graph_param_pb
    
    def serialize_compute_graph_to_file(self, file_name, name_data_map):
        graph_pb = self.serialize_compute_graph(name_data_map)
        #import ipdb; ipdb.set_trace()
        with open(file_name, "wb") as f:
            s = graph_pb.SerializeToString()
            f.write(s)
    
        
