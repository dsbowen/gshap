import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="gshap",
    version="0.0.3",
    author="Dillon Bowen",
    author_email="dsbowen@wharton.upenn.edu",
    description="A technique in explainable AI for answering broader questions in machine learning.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://dsbowen.github.io/gshap",
    packages=setuptools.find_packages(),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'numpy>=1.18.4',
        'pandas>=1.0.3',
    ],
    python_requires='>=3.6',
)