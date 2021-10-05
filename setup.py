import setuptools

DEPENDENCIES = [
    'absl-py>=0.11.0',
    'fire>=0.4.0',
    'flax',
    'gast',
    'google-cloud-storage',
    'imageio',
    'ipython',
    'jax>=0.2.7',
    'jaxlib>=0.1.69',
    'matplotlib',
    'ml_collections',
    'python-graphs>=1.1.0',
    'sklearn',
    'tensorflow',
    'tensorflow_datasets',
    'transformers>=4.6.0',
]

packages = [
    'config',
    'core',
    'core.data',
    'core.distributed',
    'core.lib',
    'core.models',
    'core.modules',
    'core.modules.ipagnn',
    'scripts',
    'third_party',
    'third_party.flax_examples',
    'experimental',
]
setuptools.setup(
    name="compressive-ipagnn",
    version="1.0.0",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    packages=packages,
    package_dir={d: d.replace('.', '/') for d in packages},
    python_requires='>=3.7',
    install_requires=DEPENDENCIES,
)
