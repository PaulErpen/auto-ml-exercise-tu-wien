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

#this takes care of the parameters step
#it also makes sure that the parameters do not stick to their bounds
#otherwise an alpha parameter for example would continually stick to the 0 or 1 boundar
#This is due to the 50/50 chance of a random value pushing the value toward its bound
#meaning once its reached the bound chances are its not gonna leave it
def get_value_step_with_unsticky(current, min, max, temperature):
    min_rand_lower = -1
    if(current == min):
        min_rand_lower = 0
    min_rand_upper = 1
    if(current == max):
        min_rand_upper = 0
    return clamp(
        current + get_random_in_range(min_rand_lower, min_rand_upper) * temperature * max,
        min,
        max)