call venv\Scripts\activate.bat
benchmark_app -m openvino_ir\resnet50.xml -data_shape [1,3,224,224] -niter 100 -d GPU