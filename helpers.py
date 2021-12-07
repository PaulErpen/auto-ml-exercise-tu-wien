from random import random as rnd

def get_random_in_range(min, max, is_int = False):
    diff = max - min
    val = rnd() * diff + min
    if(is_int):
        return int(val)
    else:
        return val

def clamp(val, min_val, max_val):
        return max(min_val, min(max_val, val))