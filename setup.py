from setuptools import setup, find_packages
from typing import List




def get_requirements(file_path:str)-> List[str]:
    """
    This function reads a requirements file and returns a list of packages.
    """
    requirements = []
    with open(file_path, 'r') as file:
        requirements = file.readlines()
        # Remove any whitespace and comments
        requirements = [req.strip() for req in requirements if req.strip() and not req.startswith('#')] 

        if '-e .' in requirements:
            requirements.remove('-e .')
    return requirements

setup(
    name='MLProject',
    version='0.1.0',
    author='Rahul Singh',
    author_email="Rahul.26gemini@gmail.com",
    packages=find_packages(),
    install_requires= get_requirements('requirements.txt')

)