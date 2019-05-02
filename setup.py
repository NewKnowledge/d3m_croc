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
      version='1.1.1',
      description='Character recognition and object classification system.',
      packages=['d3m_croc'],
      install_requires=[
                        'tensorflow-gpu <= 1.12.0',
                        'Keras == 2.2.4',
                        'pandas == 0.23.4',
                        'spacy == 2.1.0',
                        'requests >= 2.18.4, <= 2.20.0',
                        'numpy >= 1.15.4',
                        'Pillow >= 5.1.0'],
      cmdclass={
                'develop': PostDevelopCommand,
                'install': PostInstallCommand,
               }
      )
