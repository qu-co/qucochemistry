import io

from setuptools import setup, find_packages

# This reads the __version__ variable
exec(open('qucochemistry/_version.py').read())

# Readme file as long_description:
long_description = ('=================\n' +
                    'Qu & Co Chemistry\n' +
                    '=================\n')
stream = io.open('README.rst', encoding='utf-8')
stream.readline()
long_description += stream.read()


# Read in requirements.txt

setup(
    name="qucochemistry",
    version=__version__,
    author="Vincent Elfving",
    author_email="quantumcode@quandco.com",
    description="A VQE package which interfaces with Rigetti's QCS platform",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    install_requires=[
        # The minimum spec for a working qucochemistry install.
        # note to developers: this should be a subset of requirements.txt
        'openfermion==0.10.0',
        'pyquil>=2.16.0, <=2.24.0',
    ],
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    license="Apache 2",
    classifiers=[
        "Programming Language :: Python :: 3.6",
	"Programming Language :: Python :: 3.7",
	"Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
