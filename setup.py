from setuptools import setup, find_packages

version = '0.1'

setup(
    name='gan',
    version=version,
    description='Generative adversarial networks in PyTorch',
    license='MIT',
    url='http://github.com/djsaunde/gan',
    author='Daniel Saunders',
    author_email='djsaunde@cs.umass.edu',
    packages=find_packages(),
    zip_safe=False,
    install_requires=[
        'torch>=0.4.1', 'numpy', 'torchvision'
    ],
)