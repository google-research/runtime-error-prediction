NO_DATA = 'No data'
TIMEOUT = 'Timeout'

ERROR_KINDS = [
    'AssertionError',
    'AttributeError',
    # bdb.BdbQuit p02702 s706694213
    'decimal.InvalidOperation',
    'EOFError',
    'FileNotFoundError',
    'ImportError',
    'IndentationError',
    'IndexError',
    'KeyError',
    'MathDomainError',
    'MemoryError',
    'ModuleNotFoundError',
    'NameError',
    'OSError', # Bad file descriptor
    'OverflowError',
    're.error',  # nothing to repeat at position 0
    'RecursionError',
    'RuntimeError',
    'StopIteration',
    'SyntaxError',
    'TabError',
    'TypeError',
    'UnboundLocalError',
    'ValueError',
    'ZeroDivisionError',
    'numpy.AxisError',
    # 'Exception',
    # SyntaxError: invalid syntax
    # SyntaxError: invalid character
    # SyntaxError: import * only allowed at module level
    # SyntaxError: closing parenthesis
    # SyntaxError: cannot assign to operator
    # SyntaxError: Missing parentheses in call to
    # SyntaxError: from __future__ imports must occur at the beginning of the file
    # SyntaxError: invalid non-printable character
]

OTHER_ERROR = 'Other'
SILENT_ERROR = 'Silent error. Nothing in stderr.'
NO_ERROR_WITH_STDERR = 'No error (but using stderr anyway)'
NO_ERROR = 'No error'
