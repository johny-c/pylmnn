# Always prefer setuptools over distutils
from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path


if __name__ == '__main__':
    here = path.abspath(path.dirname(__file__))

    # Get the long description from the README file
    with open(path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = f.read()


    setup(
        name='PyLMNN',
        version='0.2.0',
        description='LMNN implementation in python',
        long_description=long_description,
        url='https://github.com/johny-c/pylmnn.git',
        author='John Chiotellis',
        author_email='johnyc.code@gmail.com',
        license='GPLv3',

        classifiers=[
                    'Development Status :: 3 - Alpha',
                    'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
                    'Natural Language :: English',
                    'Operating System :: MacOS :: MacOS X',
                    'Operating System :: Microsoft :: Windows',
                    'Operating System :: POSIX :: Linux',
                    'Programming Language :: Python :: 3',
                    'Programming Language :: Python :: 3.5',
                    'Topic :: Scientific/Engineering :: Artificial Intelligence'],

        packages=find_packages(exclude=['contrib', 'docs', 'tests']),
        package_dir={'pylmnn': 'pylmnn'},
        install_requires=['numpy>=1.11',
                          'scipy>=0.18',
                          'scikit_learn>=0.18',
                          'GPyOpt>=1.0.3',
                          'matplotlib>=1.5.3'],
    )
