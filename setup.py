from setuptools import setup, find_packages

with open("README.md", "r") as readme_file:
    readme = readme_file.read()

requirements = ["numpy", "pika"]

setup(
    name="pygamootest",
    version="0.0.6",
    author="Adam Marszałek & Paweł Jarosz",
    author_email="amarszalek@pk.edu.pl",
    description="GAme theory based framework for MultiObjective Optimization in Python",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/amarszalek/pyGAMOO",
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: Apache Software License",
    ],
)