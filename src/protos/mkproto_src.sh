#!/bin/bash
protoc -I=. --cpp_out=. Tensor.proto
protoc -I=. --cpp_out=. ComputeGraph.proto
mv *.pb.h ../../include/protos