import os
from setuptools import setup
from setuptools.command.develop import develop
from setuptools.command.install import install


class PostDevelopCommand(develop):
    """Post-installation for development mode."""
    def run(self):
        os.system("python3 -m spacy download en_core_web_md")
        develop.run(self)


class PostInstallCommand(install):
    """Post-installation for installation mode."""
    def run(self):
        os.system("python3 -m spacy download en_core_web_md")
        install.run(self)


setup(name='d3m_croc',
      version='1.0.0',
      description='Character recognition and object classification system.',
      packages=['nk_croc'],
      install_requires=[
                        'tensorflow == 1.8.0',
                        'Keras == 2.1.6',
                        'pandas >= 0.22.0, <= 0.23.0',
                        'tesserocr == 2.3.1',
                        'spacy == 2.0.11',
                        'requests == 2.20.0',
                        'numpy >= 1.13.3',
                        'Pillow >= 5.1.0'],
      cmdclass={
                'develop': PostDevelopCommand,
                'install': PostInstallCommand,
               }
      )
