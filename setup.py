import setuptools

packages = [
    'core', 'core.data', 'core.distributed', 'core.models.ipagnn',
    'scripts',
    'third_party', 'third_party.flax_examples',
    'experimental',
]
setuptools.setup(
    name="compressive-ipagnn",
    version="1.0.0",
    packages=packages,
    package_dir={d: d.replace('.', '/') for d in packages},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
