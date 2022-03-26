# Copyright (C) 2021 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import setuptools

DEPENDENCIES = [
    'absl-py>=0.11.0',
    'beautifulsoup4',
    'fire>=0.4.0',
    'flax',
    'gast',
    'google-cloud-storage',
    'html5lib',
    'imageio',
    'ipython',
    'jax>=0.2.7',
    'jaxlib>=0.1.69',
    'matplotlib',
    'ml_collections',
    'python-graphs>=1.2.3',
    'sklearn',
    'tensorflow',
    'tensorflow_datasets',
    'transformers>=4.6.0',
    'uTidylib',
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
    name="runtime-error-prediction",
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
