from setuptools import setup, find_packages

with open("README.md", "r") as readme_file:
    readme = readme_file.read()

#requirements = ['numpy==1.25.2', 'osbrain==0.6.5', 'pymoo==0.6.0.1', 'Pyro4==4.82', 'scipy==1.11.1',
#                'setuptools==68.2.2', 'tqdm==4.66.1']
requirements = ['numpy', 'osbrain', 'pymoo', 'Pyro4', 'scipy',
                'setuptools', 'tqdm']


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
    include_package_data=True,
    package_data={'': ['_cutils.so']},
)
