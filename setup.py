
#  fetches out all the required packages throughout the machine learning project
from setuptools import find_packages, setup

# importing list function for proper return statements for listing
from typing import List

# defining the requirements.txt file for the packages
def get_requirements(file_path: str) -> List[str]:
    '''
    :param file_path:
    :return:
    function to return a list of requirements
    '''
    requirements = []
    with open(file_path) as rq:

        requirements = rq.readlines()
        requirements = [i.replace()("\n","") for i in requirements]

        if "-e ." in requirements:
            requirements.remove( "-e ." )

    return requirements


# parameters required for further classification of the project details and packages
# mostly close to a metadata information
setup(
    name='Personal Loan',
    version='0.0.1',
    author='Ashwin_Ashok',
    author_email='ashwina506@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),
)