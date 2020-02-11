from setuptools import setup

setup(name='haven',
      version='0.2',
      description='Manage large-scale experiments',
      url='https://github.com/ElementAI/haven_borgy',
      author='Issam Laradji',
      author_email='issam.laradji@elementai.com',
      license='MIT',
      packages=['haven'],
      zip_safe=False,
      install_requires=[
        'skimage',
        'numpy',
        'gdown',
        'Pillow>=5.0.0,<7.0.0',
        'h5py~=2.9.0',
        'tqdm>=4.0.0',
        'opencv-python',
        'scopy',
        'pandas',
        'pylab'
      ]),