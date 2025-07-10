#!/bin/bash
trtexec --onnx=onnx/edgeface_xs_gamma_06.onnx --saveEngine=models/edgeface_xs.trt --fp16
