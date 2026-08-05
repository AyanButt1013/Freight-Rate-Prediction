import os
import sys

print("Current directory:", os.getcwd())
print("sys.path:")
for p in sys.path:
    print(repr(p))