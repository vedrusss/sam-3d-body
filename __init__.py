import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))

# Keep insertion order deterministic:
# 1) add sam_3d_body_fork
# 2) add sam_3d_body_fork/dinov3 at index 0 so it has higher priority
for folder in (None, "dinov3"):
    path = os.path.join(script_dir, folder) if folder else script_dir
    if os.path.isdir(path) and path not in sys.path:
        sys.path.insert(0, path)
