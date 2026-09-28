"""
The Teller's Envelopes

A bank teller has max_amount dollars in $1 bills and a stack of envelopes.
They want to fill the envelopes ahead of time so that any customer can be
handed any whole amount from $1 to max_amount without opening an envelope.
How few envelopes can they get away with?

Envelope sizes here are powers of `base`: 1, base, base**2, ...
To be able to make every amount, the teller needs (base - 1) envelopes of
each size, the same way a base-10 number can need up to 9 of each place value.
The last size gets capped at whatever money is left over.

Run this file to print a comparison of different bases.

The interactive version of this demo lives in
docs/s1_binary_envelopes/s1_binary_envelopes.html and is served at
https://chelseatroy.github.io/python-exercises/s1_binary_envelopes/s1_binary_envelopes.html
"""


def fill_envelopes(max_amount, base=2, copies_per_size=None):
    """Fill envelopes with powers of base; the last one gets whatever is left.

    copies_per_size defaults to base - 1, which is enough to make every amount.
    Pass copies_per_size=1 to see what happens when the teller skimps.
    """
    if base == 1:
        return [1] * max_amount  # every "power of 1" is 1: one envelope per dollar

    if copies_per_size is None:
        copies_per_size = base - 1

    envelopes, next_size, remaining = [], 1, max_amount
    while remaining > 0:
        for _ in range(copies_per_size):
            if remaining == 0:
                break
            envelopes.append(min(next_size, remaining))
            remaining -= envelopes[-1]
        next_size *= base
    return envelopes


def envelopes_needed(max_amount, base=2):
    """How many envelopes fill_envelopes uses for this base.

    For base 2 this is max_amount.bit_length(), and no base can do better:
    each envelope is either handed over or not, so k envelopes allow at most
    2**k different handoffs. The teller needs max_amount + 1 different
    results ($0 through max_amount), so 2**k >= max_amount + 1.
    """
    return len(fill_envelopes(max_amount, base))


def ways_to_make(envelopes):
    """ways[a] is how many different handoffs add up to exactly $a."""
    total = sum(envelopes)
    ways = [1] + [0] * total
    for envelope in envelopes:
        for amount in range(total, envelope - 1, -1):
            ways[amount] += ways[amount - envelope]
    return ways


def hand_over(amount, envelopes):
    """Pick envelopes that add up to amount, preferring the biggest ones.

    Returns None when no combination of envelopes adds up to amount.
    """
    envelopes = sorted(envelopes, reverse=True)
    # can_make[i] holds every amount reachable using only envelopes[i:]
    can_make = [set() for _ in range(len(envelopes) + 1)]
    can_make[-1] = {0}
    for i in range(len(envelopes) - 1, -1, -1):
        can_make[i] = can_make[i + 1] | {a + envelopes[i] for a in can_make[i + 1]}

    if amount not in can_make[0]:
        return None
    chosen = []
    for i, envelope in enumerate(envelopes):
        if amount - envelope in can_make[i + 1]:
            chosen.append(envelope)
            amount -= envelope
    return chosen


if __name__ == "__main__":
    max_amount = 1000
    print(f"Envelopes for any amount from $1 to ${max_amount}\n")
    print(f"{'base':>4}  {'envelopes':>9}  {'handoffs (2**k)':>16}  {'amounts':>7}  {'duplicates':>10}")
    for base in range(1, 11):
        envelopes = fill_envelopes(max_amount, base)
        k = len(envelopes)
        handoffs = 2 ** k
        amounts = max_amount + 1
        handoffs_text = f"{handoffs:,}" if k <= 40 else f"2**{k}"
        duplicates_text = f"{handoffs - amounts:,}" if k <= 40 else "a lot"
        print(f"{base:>4}  {k:>9}  {handoffs_text:>16}  {amounts:>7}  {duplicates_text:>10}")

    print("\nBase 2:", fill_envelopes(max_amount, 2))
    print("Base 3:", fill_envelopes(max_amount, 3))
    print("Base 3, one per size:", fill_envelopes(max_amount, 3, copies_per_size=1))

    skimped = fill_envelopes(max_amount, 3, copies_per_size=1)
    gaps = [a for a, w in enumerate(ways_to_make(skimped)) if w == 0]
    print(f"  ...which can't make {len(gaps)} amounts, starting with {gaps[:8]}")

    print("\nA customer asks for $700 (base 2):", hand_over(700, fill_envelopes(max_amount, 2)))
