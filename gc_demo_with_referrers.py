import gc
import sys

def demonstrate_with_referrers():
    listo = []
    listo.append(listo)

    print(f"Refcount: {sys.getrefcount(listo)}")
    print(f"Referrers: {gc.get_referrers(listo)}")
    # Shows: the list itself (circular), local namespace, and getrefcount's frame

    memory_address = id(listo)
    del listo

    # Find the object in gc
    for obj in gc.get_objects():
        if id(obj) == memory_address:
            print(f"\nFound orphaned object with circular reference")
            print(f"Referrers keeping it alive: {gc.get_referrers(obj)}")
            break

    gc.collect()

    print(f"\nAfter gc.collect():")
    print(f"Object still exists: {any(id(obj) == memory_address for obj in gc.get_objects())}")

if __name__ == "__main__":
    demonstrate_with_referrers()
