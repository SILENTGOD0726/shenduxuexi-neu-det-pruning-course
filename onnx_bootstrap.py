"""Use the optional deployment dependencies without modifying the base environment."""
import bootstrap
import os
import sys

EXTRA=bootstrap.ROOT/'.onnx_deps'
if not EXTRA.exists():raise RuntimeError('Run .venv/Scripts/python.exe setup_onnx.py first')
sys.path.insert(0,str(EXTRA))
_HANDLES=[]
if os.name=='nt':
    for relative in ['onnxruntime/capi','google/_upb']:
        path=EXTRA/relative
        if path.exists():_HANDLES.append(os.add_dll_directory(str(path)))
