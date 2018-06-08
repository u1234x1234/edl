# edl

Containerized library builds

In most of the cases https://github.com/dockcross/dockcross is used.

[NNPACK](https://github.com/Maratyszcza/NNPACK) for arm64 must be built with clang, so it uses dockcross/android-arm64, unlike dockcross/linux-arm64 which uses gcc.


Caffe2 C++ example https://github.com/jhjin/caffe2-cpp/
MXNet C api prediction https://github.com/apache/incubator-mxnet/tree/master/example/image-classification/predict-cpp

Containerized builds https://github.com/apache/incubator-mxnet/tree/master/ci
