import setuptools

packages = ['core', 'scripts', 'third_party', 'experimental']
setuptools.setup(
    name="compressive-ipagnn",
    version="1.0.0",
    packages=packages,
    package_dir={d: d for d in packages},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
