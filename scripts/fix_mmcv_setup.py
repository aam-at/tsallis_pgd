#!/usr/bin/env python3
"""Fix mmcv setup.py for Python 3.13 compatibility."""

with open("external/mmcv/setup.py", "r") as f:
    content = f.read()

old = """def get_version():
    version_file = 'mmcv/version.py'
    with open(version_file, encoding='utf-8') as f:
        exec(compile(f.read(), version_file, 'exec'))
    return locals()['__version__']"""

new = """def get_version():
    version_file = 'mmcv/version.py'
    version_ns = {}
    with open(version_file, encoding='utf-8') as f:
        exec(compile(f.read(), version_file, 'exec'), version_ns)
    return version_ns['__version__']"""

with open("external/mmcv/setup.py", "w") as f:
    f.write(content.replace(old, new))

print("Fixed setup.py for Python 3.13 compatibility")
