import os

build = os.path.join(os.path.dirname(__file__), 'build')
if not os.path.exists(build):
    os.mkdir(build)

success = os.system('cd ' + build + ' && cmake .. && make')
if (success != 0):
    raise RuntimeError('Backend cross-compile failed.')

from .build.src.backend import *
