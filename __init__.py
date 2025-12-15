import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))

for folder in {None, 'dinov3'}:
    path = os.path.join(script_dir, folder) if folder else script_dir
    if os.path.isdir(path) and path not in sys.path:
        sys.path.insert(0, path)
