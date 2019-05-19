"""
Most part of this file is borrowed from ONNX
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from distutils.spawn import find_executable
from distutils import sysconfig, log
import setuptools
import setuptools.command.build_py
import setuptools.command.develop
import setuptools.command.build_ext

from collections import namedtuple
from contextlib import contextmanager
import glob
import os
import shlex
import subprocess
import sys
import struct
from textwrap import dedent
import multiprocessing
from os.path import dirname


TOP_DIR = os.path.realpath(dirname(dirname(dirname(dirname(os.path.abspath(__file__))))))
CMAKE_BUILD_DIR = os.path.join(TOP_DIR, '.setuptools-cmake-build')

WINDOWS = (os.name == 'nt')

CMAKE = find_executable('cmake3') or find_executable('cmake')
MAKE = find_executable('make')

install_requires = ['onnx']
setup_requires = ['cmake']
tests_require = []
extras_require = {}

################################################################################
# Global variables for controlling the build variant
################################################################################

DEBUG = bool(os.getenv('DEBUG'))
COVERAGE = bool(os.getenv('COVERAGE'))

################################################################################
# Version
################################################################################

try:
    git_version = subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                          cwd=TOP_DIR).decode('ascii').strip()
except (OSError, subprocess.CalledProcessError):
    git_version = None

################################################################################
# Pre Check
################################################################################

assert CMAKE, 'Could not find "cmake" executable!'

################################################################################
# Utilities
################################################################################


@contextmanager
def cd(path):
    if not os.path.isabs(path):
        raise RuntimeError('Can only cd to absolute path, got: {}'.format(path))
    orig_path = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(orig_path)


class CMakeExtension(setuptools.Extension):
    def __init__(self, name):
        # don't invoke the original build_ext for this special extension
        super().__init__(name, sources=[])


class build_ext(setuptools.command.build_ext.build_ext):
    def run(self):
        for ext in self.extensions:
            self.build_extension(ext)


    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))

        if not os.path.exists(CMAKE_BUILD_DIR):
            os.makedirs(CMAKE_BUILD_DIR)

        with cd(CMAKE_BUILD_DIR):
            build_type = 'Release'
            # configure
            cmake_args = [
                CMAKE,
                '-DPYTHON_INCLUDE_DIR={}'.format(sysconfig.get_python_inc()),
                '-DPYTHON_EXECUTABLE={}'.format(sys.executable),
                '-DCMAKE_EXPORT_COMPILE_COMMANDS=ON',
                '-DPY_EXT_SUFFIX={}'.format(sysconfig.get_config_var('EXT_SUFFIX') or ''),
                '-DBNN_BUILD_PYTHON=ON',
                '-DBNN_SYSTEM_PROTOBUF=OFF',
                '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={}'.format(extdir),
            ]
            if COVERAGE or DEBUG:
                # in order to get accurate coverage information, the
                # build needs to turn off optimizations
                build_type = 'Debug'
            cmake_args.append('-DCMAKE_BUILD_TYPE={}'.format(build_type))
            if WINDOWS:
                cmake_args.extend([
                    # we need to link with libpython on windows, so
                    # passing python version to window in order to
                    # find python in cmake
                    '-DPY_VERSION={}'.format('{0}.{1}'.format(*sys.version_info[:2])),
                    '-DBNN_USE_MSVC_STATIC_RUNTIME=ON',
                ])
                if 8 * struct.calcsize("P") == 64:
                    # Temp fix for CI
                    # TODO: need a better way to determine generator
                    cmake_args.append('-DCMAKE_GENERATOR_PLATFORM=x64')
            if 'CMAKE_ARGS' in os.environ:
                extra_cmake_args = shlex.split(os.environ['CMAKE_ARGS'])
                # prevent crossfire with downstream scripts
                del os.environ['CMAKE_ARGS']
                log.info('Extra cmake args: {}'.format(extra_cmake_args))
                cmake_args.extend(extra_cmake_args)
            cmake_args.append(TOP_DIR)
            subprocess.check_call(cmake_args)

            build_args = [CMAKE, '--build', os.curdir]
            if WINDOWS:
                build_args.extend(['--config', build_type])
                build_args.extend(['--', '/maxcpucount:{}'.format(multiprocessing.cpu_count())])
            else:
                build_args.extend(['--', '-j', str(multiprocessing.cpu_count())])
            subprocess.check_call(build_args)


cmdclass = {
    'build_ext': build_ext,
}

################################################################################
# Extensions
################################################################################

ext_modules = [
    CMakeExtension("_onnx2bnn")
]

################################################################################
# Packages
################################################################################

# no need to do fancy stuff so far
packages = setuptools.find_packages()

################################################################################
# Final
################################################################################

setuptools.setup(
    name="onnx2bnn",
    # version=VersionInfo.version,
    description="Convert ONNX to dabnn",
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    packages=packages,
    include_package_data=True,
    install_requires=install_requires,
    setup_requires=setup_requires,
    tests_require=tests_require,
    extras_require=extras_require,
    author='daquexian',
    author_email='daquexian566@gmail.com',
    url='https://github.com/JDAI-CV/dabnn',
)


