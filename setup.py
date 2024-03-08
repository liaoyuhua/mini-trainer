import setuptools
import mini_trainer

with open("README.md", "r") as fh:
    long_description = fh.read()

install_requires = [line.strip() for line in open("requirements.txt").readlines()]

setuptools.setup(
    name="mini_trainer",
    version=mini_trainer.__version__,
    author="liaoyuhua",
    author_email="ml.liaoyuhua@gmail.com",
    description="The PyTorch micro framework for mini_trainer neural networks.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/liaoyuhua/mini-trainer",
    packages=setuptools.find_packages(),
    install_requires=install_requires,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',
)
