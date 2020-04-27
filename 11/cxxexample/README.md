cxxexample README

1. 下载opencv-4.1.0源代码并解压至``3rdparty/``
2. ``cd 3rdparty; bash build_opencv.sh``
3. 下载insightface检测、识别预训练模型并解压到``asserts/``
4. 编译cxx代码: ``python setup.py build_ext -i`` to compile cxx lib
5. 打包未加密程序包: ``python make_unencrypted_package.py``

