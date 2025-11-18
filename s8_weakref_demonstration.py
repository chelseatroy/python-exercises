import weakref
import sys
import gc

print("The Problem - Circular References")
print("=====================================")

class Node:
    """A linked list node that creates circular references"""
    def __init__(self, value):
        self.value = value
        self.next = None

    def __repr__(self):
        return f"Node({self.value})"

# Create a cycle
a = Node(1)
b = Node(2)
a.next = b
b.next = a

print(f"\nCreated two nodes with circular reference:")
print(f"Node a: {a}, refcount: {sys.getrefcount(a)}")
print(f"Node b: {b}, refcount: {sys.getrefcount(b)}")

# Save addresses to check later
addr_a = id(a)
addr_b = id(b)

# Delete the names
del a, b

print(f"\nAfter deleting names, checking if objects still exist:")
objects_exist = any(id(obj) in [addr_a, addr_b] for obj in gc.get_objects())
print(f"Objects still in memory: {objects_exist}")

# Clean up with GC
collected = gc.collect()
print(f"\nGC collected {collected} objects")
objects_exist = any(id(obj) in [addr_a, addr_b] for obj in gc.get_objects())
print(f"Objects still in memory after GC: {objects_exist}")

print("PART 2: The Solution - Using weakref")
print("=====================================")

class SmartNode:
    """A node that uses weakref to prevent cycles"""
    def __init__(self, value):
        self.value = value
        self._next = None  # Will store a weakref

    @property
    def next(self):
        """Get the next node (dereference the weakref)"""
        if self._next is None:
            return None
        return self._next()

    @next.setter
    def next(self, node):
        """Set the next node (store as weakref)"""
        if node is None:
            self._next = None
        else:
            self._next = weakref.ref(node)

    def __repr__(self):
        return f"SmartNode({self.value})"

# Create the same structure with SmartNode
c = SmartNode(1)
d = SmartNode(2)
c.next = d
d.next = c  # This creates a weakref

print(f"\nCreated two SmartNodes:")
print(f"Node c: {c}, refcount: {sys.getrefcount(c)}")
print(f"Node d: {d}, refcount: {sys.getrefcount(d)}")
print(f"c.next points to: {c.next}")
print(f"d.next points to: {d.next}")

# Save addresses
addr_c = id(c)
addr_d = id(d)

# Delete just one name
print(f"\nDeleting 'd' (but c.next still 'points' to it via weakref):")
del d

# Try to access through weakref
print(f"c.next is now: {c.next}")  # Should be None; the object was freed.

# Delete the other
del c

print(f"\nAfter deleting both names:")
objects_exist = any(id(obj) in [addr_c, addr_d] for obj in gc.get_objects())
print(f"Objects still in memory: {objects_exist}")