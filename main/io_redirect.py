import os
import sys
from . import state

class StdoutRedirector:
    def __init__(self, writer):
        self.writer = writer
        self.terminal = sys.__stdout__
    def write(self, message):
        try:
            self.terminal.write(message)
            self.terminal.flush()
            if self.writer:
                try:
                    self.writer.write(message)
                    self.writer.flush()
                except Exception:
                    self.writer = None
        except Exception:
            pass
    def flush(self):
        self.terminal.flush()

def suppress_stdout():
    if not state.saved_stdout and not state.saved_stderr and not state.saved_stdout_py and not state.saved_stderr_py and not state.stdout_nullfile and not state.stdout_nullfile_py:
        sys.stdout.flush()
        sys.stderr.flush()
        state.saved_stdout = os.dup(sys.stdout.fileno())
        state.saved_stderr = os.dup(sys.stderr.fileno())
        state.saved_stderr_py = sys.stderr
        state.saved_stdout_py = sys.stdout
        state.stdout_nullfile = os.open(os.devnull, os.O_WRONLY)
        state.stdout_nullfile_py = open(os.devnull, 'w')
        os.dup2(state.stdout_nullfile, sys.stdout.fileno())
        os.dup2(state.stdout_nullfile, sys.stderr.fileno())
        sys.stderr = sys.stdout = state.stdout_nullfile_py

def restore_stdout():
    if state.saved_stdout and state.saved_stderr and state.saved_stdout_py and state.saved_stderr_py and state.stdout_nullfile and state.stdout_nullfile_py:
        sys.stdout = state.saved_stdout_py
        sys.stderr = state.saved_stderr_py
        os.dup2(state.saved_stdout, sys.stdout.fileno())
        os.dup2(state.saved_stderr, sys.stderr.fileno())
        os.close(state.stdout_nullfile)
        state.stdout_nullfile_py.close()
        os.close(state.saved_stdout)
        os.close(state.saved_stderr)
        state.saved_stdout = state.saved_stderr = state.saved_stdout_py = state.saved_stderr_py = state.stdout_nullfile = state.stdout_nullfile_py = None
