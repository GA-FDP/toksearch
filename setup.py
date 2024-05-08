# Copyright 2024 General Atomics
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#    http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from setuptools import setup
from setuptools import Extension
from setuptools import find_packages
import os


from pathlib import Path
root_path = Path(__file__).parent
version_file = root_path / "toksearch" / "VERSION"


def ell1_extension():
    # Check if the user has specified an include dir for cblas.h. 
    # If not, then check whether we're using a conda env
    # (presumably with openblas installed)
    # Failing that, do nothing and hope for the best.
    import numpy as np

    conda_prefix = os.getenv('CONDA_PREFIX', None)
    user_blas_dir = os.getenv('ELL1_BLAS_INCLUDE_DIR', None)

    if user_blas_dir:
        include_dirs = [user_blas_dir]
    elif conda_prefix:
        blas_include_dir = os.path.join(conda_prefix, 'include')
        include_dirs = [blas_include_dir]
    else:
        include_dirs = []

    include_dirs += [np.get_include(), 'src',]

    return Extension(
        'toksearch.library.ell1module',
        sources=['src/ell1_python.c', 'src/ell1lib.c'],
        libraries=[
            'm',
            'rt',
            'pthread',
            'gfortran',
            'openblas',
        ],
        extra_compile_args=['-Wno-unused-variable', '-O0', '-g', '-std=c99'],
        include_dirs=include_dirs,
    )

try:
    extensions = [ell1_extension(),]
except ImportError:
    extensions = []

packages = ["toksearch",]
packages += [os.path.join('toksearch', package) for package in find_packages('toksearch')]

setup(
    name = 'toksearch',
    setup_requires=["setuptools-git-versioning>=2.0,<3"],
    setuptools_git_versioning={
        "enabled": True,
        "version_file": version_file,
        "count_commits_from_version_file": True,  # <--- enable commits tracking
        "dev_template": "{tag}.dev{ccount}",  # suffix for versions will be .dev
        "dirty_template": "{tag}.dev{ccount}",  # same thing here
    },
    ext_modules = extensions,
    include_package_data=True,
    packages=packages,
    scripts=['scripts/toksearch_submit', 'scripts/toksearch_shape', 'scripts/toksearch_example.py'],
    # this package will read some included files in runtime, avoid installing it as .zip
    zip_safe=False,
      
)
