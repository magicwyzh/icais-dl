#!/bin/bash
export LD_LIBRARY_PATH=../../deps/lib:$LD_LIBRARY_PATH
# Create Cpp source
../../deps/bin/protoc -I=. --cpp_out=. Tensor.proto
../../deps/bin/protoc -I=. --cpp_out=. ComputeGraph.proto
mv *.pb.h ../../include/protos
# Create Python source
../../deps/bin/protoc -I=. --python_out=. ./*.proto
mv *.py ../../python/icdl/

