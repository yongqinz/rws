import os
import sys
from setuptools import setup, Extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(BuildExtension):
    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = [
            '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
            '-DPYTHON_EXECUTABLE=' + sys.executable,
        ]

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        import subprocess
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp)
        subprocess.check_call(['cmake', '--build', '.', '--target', 'mxmoe_ops'], cwd=self.build_temp)

setup(
    name='mxmoe_ops',
    ext_modules=[CMakeExtension('mxmoe_ops')],
    cmdclass={
        'build_ext': CMakeBuild,
    }
)