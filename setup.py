from setuptools import setup
from setuptools import find_packages

setup(name='fasttext-langdetect',
      version='1.0.6',
      description='80x faster and 95% accurate language identification with Fasttext',
      keywords=['fasttext', 'langdetect', 'language detection',
                'language identification'],
      long_description=open("README.md", "r", encoding='utf-8').read(),
      long_description_content_type="text/markdown",
      author='Zafer Cavdar',
      author_email='zafercavdar@yahoo.com',
      install_requires=[
          "fasttext @ git+https://github.com/shivdeepak/fastText#9bacdb9525447e6b564598a3f8b7e5ed0eb2eba5",
          "requests>=2.22.0",
      ],
      license='MIT',
      packages=find_packages(),
      classifiers=[
          'Development Status :: 5 - Production/Stable',
          'Intended Audience :: Developers',
          'Topic :: Scientific/Engineering :: Artificial Intelligence',
          'Topic :: Scientific/Engineering :: Information Analysis',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          'Programming Language :: Python :: 3.8',
          'Programming Language :: Python :: 3.9',
      ])
