import setuptools

setuptools.setup(
    name="compressive-ipagnn",
    version="1.0.0",
    packages=['core', 'scripts', 'third_party', 'experimental'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
