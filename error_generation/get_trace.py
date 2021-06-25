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


import json
import sys
from collections import defaultdict
import subprocess


def postprocess_and_save(json_fp, offset, processed_suffix):
    """Here we offset the lines of the trace to take into account
    additional lines added to get the trace of the function.
    """
    data = json.load(open(json_fp, "rb"))
    processed_data = {}
    for key, val in data.items():
        val = [v for v in val if v]
        if val:
            processed_data[int(key) - offset] = val
    out_path = json_fp.replace(".json", "_" + str(processed_suffix) + ".json")
    open(out_path, "w").write(json.dumps(processed_data))


def run_for_errors(
    python_filepath,
    data_path,
    trace_path,
    stdin_file,
    stdout_file,
    stderr_file,
    processed_suffix="processed",
    offset=3,
):
    # Assumes the input is stdin when called.
    # import pdb;pdb.set_trace()
    trace_source = open(trace_path, "r").read()
    python_source = open(python_filepath, "r").read()
    python_source = python_source.replace('__name__ == "__main__"', "True")
    python_source = python_source.replace("__name__ == '__main__'", "True")
    # TODO(rishab): Clean the python_source variable.
    python_source = (
        "import json\n"
        + "import sys\n"
        + "def main__errorchecker__():\n"
        + "\n".join("  " + line for line in python_source.split("\n"))
        + "\n"
        + trace_source
        + "\nsys.settrace(trace_calls)\n"
        + "main__errorchecker__()\n"
        + 'open("'
        + data_path
        + '","w").write(json.dumps(data, indent=4, sort_keys=True))\n'
    )
    try:
        subprocess_call = subprocess.check_call(
            ["python", "-c", python_source],
            stdin=open(stdin_file, "rb"),
            stdout=open(stdout_file, "wb"),
            stderr=open(stderr_file, "wb"),
        )
    except Exception as e:
        # raise e
        return False
    postprocess_and_save(data_path, offset, processed_suffix)
    return True
