from collections import defaultdict

data = defaultdict(list)

def trace_lines(frame, event, arg):
    if event != 'line':
        return
    co = frame.f_code
    func_name = co.co_name
    line_no = frame.f_lineno
    filename = co.co_filename
    if filename=="<string>":
        locals_dict = {}
        for key, value in frame.f_locals.items():
            try:
                json.dumps(value)
                locals_dict[key] = value
            except Exception as e:
                _ = ""
        data[line_no].append(locals_dict)


def trace_calls(frame, event, arg):
    if event!="call":
        return
    co = frame.f_code
    func_name = co.co_name
    return trace_lines
