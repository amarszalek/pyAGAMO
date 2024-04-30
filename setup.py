from setuptools import setup, find_packages

with open("README.rst", "r") as readme_file:
    readme = readme_file.read()

requirements = ["numpy", "pika"]

setup(
    name="pyagamo",
    version="0.0.10",
    author="Adam Marszałek & Paweł Jarosz",
    author_email="amarszalek@pk.edu.pl",
    description="Asynchronous GAme theory based framework for MultiObjective Optimization in Python",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/amarszalek/pyAGAMO",
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: Apache Software License",
    ],
)