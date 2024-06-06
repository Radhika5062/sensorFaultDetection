from setuptools import find_packages, setup 
from typing import List 

# Find packages searches our full project and determines which are packages. The ones that are packages will have
# init file. 

# Now our install_packages is a list with names of all the packages that need to be installed. However, if there are
# a lot of libraries that need to be installed then this list will be too long. To fix this, we are going to create
# the function below. 
def get_requirements()-> List[str]:
    requirements_list: List[str] = [] 
    return requirements_list

setup(
    name = 'sensor',
    version = '0.0.1',
    author = 'Radhika',
    author_email='radhikamaheshwari26@gmail.com',
    packages = find_packages(),
    # Below line mentions that you need to install these things. If there are more than one items then
    # you need to update this list. 
    install_requires = get_requirements(),  #['pymongo']
)