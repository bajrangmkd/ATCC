# diag_import.py (put in D:\ATCC\atcc_service)
import importlib, traceback, sys

print("Python:", sys.version)
print("CWD:", __import__("os").getcwd())

try:
    m = importlib.import_module("app.main")
    print("import succeeded")
    print("has attribute 'app':", hasattr(m, "app"))
    print("interesting attrs:", [n for n in dir(m) if n == "app" or n.startswith("load_")][:50])
except Exception:
    print("IMPORT ERROR (full traceback):")
    traceback.print_exc()
