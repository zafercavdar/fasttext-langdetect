from setuptools import setup
from setuptools import find_packages

setup(name='fasttext-langdetect',
      version='1.0.5',
      description='80x faster and 95% accurate language identification with Fasttext',
      keywords=['fasttext', 'langdetect', 'language detection',
                'language identification'],
      long_description=open("README.md", "r", encoding='utf-8').read(),
      long_description_content_type="text/markdown",
      url='https://github.com/zafercavdar/fasttext-langdetect.git',
      download_url='https://github.com/zafercavdar/fasttext-langdetect/archive/refs/tags/v1.0.5.tar.gz',
      author='Zafer Cavdar',
      author_email='zafercavdar@yahoo.com',
      install_requires=[
          "fasttext>=0.9.1",
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
