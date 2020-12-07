from setuptools import setup
import pathlib

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

setup(
    name='FastMapy',
    version='0.0.1',
    packages=['utils', 'fastmap', 'fastmap.distances'],
    url='https://github.com/shawn-davis/FastMapy',
    license='GPL-3.0 License',
    author='Shawn Davis',
    author_email='shawndavis.lomod@gmail.com',
    description='Python Implementation of the FastMap MDS technique.',
    long_description=README,
    long_description_content_type="text/markdown",
    install_requires=[
        'numpy',
        'pebble'
    ]
)
