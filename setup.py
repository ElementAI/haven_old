from setuptools import setup

setup(name='havenai',
      version='0.6.0',
      description='Manage large-scale experiments',
      url='https://github.com/ElementAI/haven',
      maintainer='Issam Laradji',
      maintainer_email='issam.laradji@elementai.com',
      license='MIT',
      packages=['haven'],
      zip_safe=False,
      install_requires=[
        'tqdm>=4.42.0'
        'matplotlib>=3.1.2',
        'numpy>=1.17.4',
        'opencv-python-headless>=4.1.2.30',
        'pandas>=0.25.3',
        'Pillow>=6.1',
        'scikit-image>=0.16.2',
        'scikit-learn>=0.22',
        'scipy>=1.3.1',
        'sklearn>=0.0',
        'torch>=0.0',
        'torchvision>=0.0',
        'notebook >= 4.0'
      ]),