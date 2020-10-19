'''
python setup.py build_ext -i
to compile
'''

# setup.py
from distutils.core import setup, Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import numpy
import os
import glob


sources = list(glob.glob("source/*.cc"))
opencv_dir = "./3rdparty/install"
# opencv_dir = r'/home/iss/wsp/FAT/11/cxxexample/3rdparty/install'
include_dirs=[numpy.get_include(), os.path.join(opencv_dir, "include", "opencv4")]

orders = " -lopencv_dnn -lopencv_gapi -lopencv_ml -lopencv_objdetect -lopencv_photo -lopencv_stitching -lopencv_video -lopencv_calib3d -lopencv_features2d -lopencv_highgui -lopencv_flann -lopencv_videoio -lopencv_imgcodecs -lopencv_imgproc -lopencv_core"

orders = orders.split(' -l')
ordermap = {}

iorder = 0
for order in orders:
    order = order.strip()
    if len(order)==0:
        continue
    ordermap[order] = iorder
    iorder+=1

print(ordermap)
extra_objects = [None] * len(ordermap)
opencv_lib_dir = os.path.join(opencv_dir, 'lib64')
if not os.path.exists(opencv_lib_dir):
    opencv_lib_dir = os.path.join(opencv_dir, 'lib')

for _file in glob.glob("%s/*.a"%opencv_lib_dir):
    name = _file.split('/')[-1][3:-2]
    idx = ordermap[name]
    extra_objects[idx] = _file
    #extra_objects.append(_file)
print(extra_objects)
for _file in glob.glob("%s/opencv4/3rdparty/*.a"%opencv_lib_dir):
    extra_objects.append(_file)

setup(
	name = 'pyfat',
    cmdclass={'build_ext': build_ext},
    ext_modules=[Extension("pyfat_implement",
                 sources=["pyfat_implement.pyx"]+sources,
                 language='c++',
                 extra_compile_args=["-std=c++11"],
                 include_dirs=include_dirs,
                 extra_objects=extra_objects
                 )],
)

