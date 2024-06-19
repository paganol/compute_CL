from setuptools import find_packages, setup

setup(
    name="compute_CL",
    version="0.1",
    description="Computes the confidence levels given a 1D posterior",
    zip_safe=False,
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "scipy",
        "dataclasses",
    ],
)
