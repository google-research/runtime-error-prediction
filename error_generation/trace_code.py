# Get line by line trace for variables in a program.
# It is adapted from the examples in the following link:
# https://pymotw.com/2/sys/tracing.html

from collections import defaultdict

data = defaultdict(list)


def trace_lines(frame, event, arg):
    if event != "line":
        return
    co = frame.f_code
    # func_name = co.co_name
    line_no = frame.f_lineno
    filename = co.co_filename
    if filename == "<string>":
        local_data_dict = {}
        for key, value in frame.f_locals.items():
            try:
                json.dumps(value)
                local_data_dict[key] = value
            except Exception as e:
                continue
        data[line_no].append(local_data_dict)


def trace_calls(frame, event, arg):
    if event != "call":
        return
    # co = frame.f_code
    # func_name = co.co_name
    return trace_lines
