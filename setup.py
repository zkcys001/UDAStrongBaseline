from setuptools import setup, find_packages


setup(name='UDAreid',
      version='1.1.0',
      description='Open-source stronger baseline for unsupervised or domain adaptive object re-ID.',
      author='Kecheng Zheng',
      author_email='zkcys001@mail.ustc.edu.cn',
      url='https://github.com/zkcys001/UDAStrongBaseline.git',
      install_requires=[
          'numpy', 'torch', 'torchvision', 
          'six', 'h5py', 'Pillow', 'scipy',
          'scikit-learn', 'metric-learn'],
      packages=find_packages(),
      keywords=[
          'Unsupervised Domain Adaptation',
          'Person Re-identification',
          'Deep Learning',
      ])
