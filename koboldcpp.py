#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# KoboldCpp compatibility shim.
# All implementation has been moved to the main/ package.
# Names are resolved lazily so that 'from koboldcpp import X' still works,
# and running 'python koboldcpp.py' delegates cleanly to main/entry.py.

_EXPORT_MAP = {
    # GUI
    'show_gui':           'main.gui',
    'show_gui_msgbox':    'main.gui',
    'show_gui_yesnobox':  'main.gui',
    # HTTP server
    'KcppProxyHandler':         'main.server',
    'KcppProxyHttpServer':      'main.server',
    'run_router_proxy':         'main.server',
    'KcppServerRequestHandler': 'main.server',
    'RunServerMultiThreaded':   'main.server',
    # Entry-point / process management
    'setuptunnel':               'main.entry',
    'register_koboldcpp':        'main.entry',
    'unregister_koboldcpp':      'main.entry',
    'main':                      'main.entry',
    'kcpp_main_process':         'main.entry',
    'mk_lora_info':              'main.entry',
    'disableSwappedFieldsInConfig': 'main.entry',
    # State module (for callers that do koboldcpp.state.X)
    'state': 'main.state',
}


def __getattr__(name):
    """Lazy resolution of public names for backward-compat imports."""
    import importlib
    if name in _EXPORT_MAP:
        mod_name = _EXPORT_MAP[name]
        mod = importlib.import_module(mod_name)
        # For the state module itself, return the module object
        if mod_name == 'main.state' and name == 'state':
            return mod
        return getattr(mod, name)
    raise AttributeError(f"module 'koboldcpp' has no attribute {name!r}")


if __name__ == '__main__':
    import runpy
    # main.entry has the full argparse + main() CLI block.
    # alter_sys=True updates sys.argv[0] to point at entry.py.
    runpy.run_module('main.entry', run_name='__main__', alter_sys=True)
