import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mini-trainer",
    version="0.1.0",
    author="Yuhua Liao",
    author_email="ml.liaoyuhua@gmail.com",
    description="A mini Trainer for PyTorch ecosystem.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/liaoyuhua/mini-trainer",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)