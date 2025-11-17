import sys
import ctypes

simple_list = [1, 2, 3]
addr = id(simple_list)
print(f"Refcount: {sys.getrefcount(simple_list)}")

del simple_list
print(f"After del: {ctypes.c_long.from_address(addr)}")  # Likely 0 or reused
