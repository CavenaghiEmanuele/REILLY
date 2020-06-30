import os
import time

build = os.path.join(os.path.dirname(__file__), 'build')
if not os.path.exists(build):
    os.mkdir(build)

success = os.system('cd ' + build + ' && cmake .. && make')
if (success != 0):
    raise RuntimeError('Backend cross-compile failed.')

# Wait for cache-to-disk flushing
time.sleep(1.0)

from .build.src.backend import *
