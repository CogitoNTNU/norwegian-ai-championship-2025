import random

rng = None  # Global random number generator instance

def seed(seed_value: str):
    """
    Seed the random number generator with the given seed value.

    :param seed_value: The seed value as a string.
    """
    global rng
    rng = random.Random(seed_value)
    print(f"Seeded RNG with {seed_value}")
    print(rng)

def random_choice(arr):
    return rng.choice(arr)

def random_number():
    """
    Generate a random number using the seeded RNG.

    :return: A random float between 0 and 1.
    :raises: RuntimeError if the RNG has not been seeded.
    """
    if rng is None:
        raise RuntimeError("RNG not seeded.")
    return rng.random()