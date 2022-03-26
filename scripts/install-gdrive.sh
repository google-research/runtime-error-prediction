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

wget 'https://docs.google.com/uc?id=0B3X9GlR6EmbnWksyTEtCM0VfaFE&export=download'
mv uc\?id\=0B3X9GlR6EmbnWksyTEtCM0VfaFE\&export\=download gdrive
chmod +x gdrive
sudo install gdrive /usr/local/bin/gdrive

# To install and grant permissions:
# gdrive list

# To download just full-noudf-ids:
# gdrive download --recursive 1UEHe18x3BnvQy8cTz1JfEvS-eEe68120

# Older datasets:
# To download just full-noudf:
# gdrive download --recursive 1O3rEKL6k2pwAU5xnyVEMH0B_jXfEk9Dc
# To download all project-codenet-data:
# gdrive download --recursive 12Ji3yan98_4U1dVxY3PPyhVa4uHZLc6J
