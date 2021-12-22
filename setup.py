from setuptools import setup


setup(name='photometry',
      version='0.0.1',
      author='Arnaud de Mattia',
      author_email='',
      description='Package for photometric analyses',
      license='GPL3',
      url='http://github.com/adematti/photometry',
      install_requires=['numpy', 'scipy', 'matplotlib', 'fitsio', 'mpi4py', 'healpy'],
      packages=['photometry']
)
