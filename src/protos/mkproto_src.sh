#!/bin/bash
# Create Cpp source
protoc -I=. --cpp_out=. Tensor.proto
protoc -I=. --cpp_out=. ComputeGraph.proto
mv *.pb.h ../../include/protos
# Create Python source
protoc -I=. --python_out=. ./*.proto
mv *.py ../../python/icdl/

