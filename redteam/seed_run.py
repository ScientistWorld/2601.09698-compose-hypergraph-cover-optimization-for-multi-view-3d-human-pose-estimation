import random
import runpy
import sys
import numpy as np
seed = int(sys.argv[1])
random.seed(seed)
np.random.seed(seed)
sys.argv = ['method/compose.py', *sys.argv[2:]]
runpy.run_path('method/compose.py', run_name='__main__')
