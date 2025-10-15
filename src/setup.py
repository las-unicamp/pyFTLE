import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup

extensions = [
    Extension(
        name="interp3d.interp",
        sources=["interp3d/interp.pyx"],
        include_dirs=[np.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    ),
    Extension(
        name="cython_integrators.integrators",
        sources=["cython_integrators/integrators.pyx"],
        include_dirs=[np.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    ),
]

setup(
    name="pyFTLE",
    version="0.1",
    package_dir={"": "."},  # or {"": "src"} if running setup from project root
    packages=["interp3d", "cython_integrators"],
    ext_modules=cythonize(extensions, language_level="3"),
)
