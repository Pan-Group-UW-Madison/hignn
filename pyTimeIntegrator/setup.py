from setuptools import setup, Extension

functions_module = Extension (
    name='pyTimeIntegrator',
    sources=['pyTimeIntegrator.cpp'],
    include_dirs=[r'$(python3 -m pybind11 --includes)', r'.\\'],
    extra_compile_args = ["-fopenmp"]
)

setup(ext_modules=[functions_module])