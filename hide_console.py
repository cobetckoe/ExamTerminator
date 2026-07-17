"""PyInstaller运行时钩子：隐藏subprocess黑窗"""
import sys
import subprocess

if sys.platform == "win32":
    _original_popen_init = subprocess.Popen.__init__
    
    def _patched_popen_init(self, *args, **kwargs):
        if 'creationflags' not in kwargs:
            kwargs['creationflags'] = subprocess.CREATE_NO_WINDOW
        _original_popen_init(self, *args, **kwargs)
    
    subprocess.Popen.__init__ = _patched_popen_init
