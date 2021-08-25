import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="simeq-ipozdeev",
    version="0.0.1",
    author="Igor Pozdeev",
    author_email="igor.pozdeev@unisg.ch",
    description="Simultaneous equations build-up and preprocessing for "
                "estimation.",
    long_description="Simultaneous equations build-up and preprocessing for "
                "estimation.",
    long_description_content_type="text/markdown",
    url="https://github.com/ipozdeev/simeq/",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.5',
)