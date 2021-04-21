import os
import json

__BACKEND__ = 'pyg'

def _get_path():
    return os.path.join(os.path.expanduser('~'), '.autogl', 'backend.json')

def _check_backend():
    global __BACKEND__
    path = _get_path()
    if 'AUTOGL_BACKEND' in os.environ:
        __BACKEND__ = os.environ['AUTOGL_BACKEND']
    elif os.path.exists(_get_path()):
        __BACKEND__ = json.load(open(path, 'r'))['backend']
    else:
        try:
            import torch_geometric
            __BACKEND__ = 'pyg'
        except ImportError:
            try:
                import dgl
                __BACKEND__ = 'dgl'
            except ImportError as error:
                raise ImportError("Cannot find suitable backend, please install pytorch_geometric or dgl first!") from error
    __BACKEND__ = __BACKEND__.lower()
    assert __BACKEND__ in ["dgl", "pyg"], "Currently only support backend [dgl, pyg], but get {} instead".format(__BACKEND__)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump({'backend': __BACKEND__}, open(path, 'w'))
    return __BACKEND__

__all__ = ["_check_backend", "_get_path"]