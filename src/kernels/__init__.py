import sys, os
_kernels_dir = os.path.dirname(os.path.abspath(__file__))
if _kernels_dir not in sys.path:
    sys.path.insert(0, _kernels_dir)
from pyre_kernels import *
