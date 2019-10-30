from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np
extensions=[
    Extension("pixel_link_decode",
             ["pixel_link_decode.pyx"],
             # include_dirs=[numpy.get_include()],
              include_dirs=[np.get_include()]
            )
]
setup(
    ext_modules=cythonize(extensions)
)