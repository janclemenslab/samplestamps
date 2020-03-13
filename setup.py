from setuptools import setup, find_packages
import os

# read the contents of your README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(name='sample stamps',
      version='0.2.0',
      description='samplestamps',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url='http://github.com/janclemenslab/samplestamps',
      author='Jan Clemens',
      author_email='clemensjan@googlemail.com',
      license='MIT',
      packages=find_packages('src'),
      package_dir={'': 'src'},
      install_requires=['numpy'],
      include_package_data=True,
      zip_safe=False
      )
