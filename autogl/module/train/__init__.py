from ...backend import __BACKEND__

if __BACKEND__ == 'pyg':
    from .pyg import *
elif __BACKEND__ == 'dgl':
    from .dgl import *