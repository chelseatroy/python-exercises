"""
## Challenge example!

It simulates a mini workload that combines:

1. Async I/O (asyncio.sleep): pretending to fetch data from multiple URLs or APIs.
2. Multiprocessing (ProcessPoolExecutor): doing real CPU-bound work in parallel (prime counting).
3. Threading (threading.Thread): running a lightweight progress reporter that updates while other tasks run.

Take a look at this code:
"""
import asyncio, math, time, threading, os
from concurrent.futures import ProcessPoolExecutor

def is_prime(n: int) -> bool:
    if n < 2: return False
    if n % 2 == 0: return n == 2
    r = int(math.isqrt(n))
    for f in range(3, r+1, 2):
        if n % f == 0: return False
    return True

def cpu_task(limit: int) -> int:
    return sum(1 for x in range(limit-5000, limit) if is_prime(x))

async def run():
    start = time.time()
    progress, stop = [0], threading.Event()

    def reporter():
        while not stop.is_set():
            print(f"progress: {progress[0]}/20", end="\r")
            time.sleep(0.2)

    t = threading.Thread(target=reporter, daemon=True); t.start()
    loop = asyncio.get_running_loop()
    nums = [200_000 + i*10_000 for i in range(20)]

    with ProcessPoolExecutor(max_workers=3) as ex:
        async def one(n):
            await asyncio.sleep(0.1)                  # fake async I/O
            res = await loop.run_in_executor(ex, cpu_task, n)
            progress[0] += 1
            return res
        results = await asyncio.gather(*(one(n) for n in nums))

    stop.set(); t.join()
    print(f"\nprimes counted: {sum(results)} in {time.time()-start:.2f}s")

if __name__ == "__main__":
    # Required on Windows/macOS (spawn). Safe everywhere.
    # Optional: from multiprocessing import freeze_support; freeze_support()
    asyncio.run(run())

"""
QUESTIONS: 

1. Why do you think each portion of the code uses the concurrency method that it uses? 
2. Why does the progress counter sometimes not reach 20? 
Is that because not all of the primes are being counted? 
If not, how can we fix the progress counter to show the actual progress?

3.Try any of the following and witness the effects on the program:

Remove multiprocessing → watch it slow down (one core).
Remove async sleep → see pure CPU parallelism.
Change ProcessPoolExecutor → ThreadPoolExecutor → see GIL effects.
Change nums length or window size → control load and observe scaling.

DO NOT set the number of cores above 4, please. 
"""