import sys
import ctypes
import gc

circular_list = []
circular_list.append(circular_list)
addr = id(circular_list)
print(f"Refcount: {sys.getrefcount(circular_list)}")

del circular_list
print(f"After del, before GC: {ctypes.c_long.from_address(addr)}")  # Still has refcount

gc.collect()
print(f"After gc.collect(): {ctypes.c_long.from_address(addr)}")  # Should be 0
