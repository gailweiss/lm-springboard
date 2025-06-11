import random
import string
from copy import deepcopy
from data.support import RawSample


class SyntheticSamplesIterator:
    def __init__(self, generator, size):
        self.generator = generator
        self.size = size

    def __getitem__(self, i):
        if i >= self.size:
            raise IndexError
        s = random.getstate()
        random.seed(i)
        res = self.generator()
        random.setstate(s)
        if isinstance(res, str):
            res = RawSample(res)
        return res

    def __len__(self):
        return self.size

    def cropped(self, crop):
        res = deepcopy(self)
        res.size = crop
        return res


class SyntheticData:
    def __init__(self):
        self.generators = {}
        self.sizes = {}

    def names(self):
        return list(self.generators.keys())

    def has_dataset(self, name):
        return name in self.names()

    def get(self, name):
        f = self.generators[name]
        # as of 2024.09.11 : random seed choice has changed,
        # comparison of new loss curves to existing ones not reliable
        return SyntheticSamplesIterator(f, self.sizes[name])

    def register(self, name, f, size):
        self.generators[name] = f
        self.sizes[name] = int(size)


default_size = 1e6
syntheticdatasets = SyntheticData()


def registered(n=default_size, name=None):
    def _registered(f):
        rname = f.__name__ if None is name else name
        syntheticdatasets.register(rname, f, n)
        return f
    return _registered


@registered()
def histogram():
    vocab = string.ascii_lowercase + string.ascii_uppercase
    n = random.randint(10, 80)
    letters = ''.join(random.choices(vocab, k=n))
    res = [letters.count(t) for t in letters]
    return letters + "::" + str(res)


@registered()
def long_addition():
    n1 = random.randint(2, 20)
    n2 = random.randint(2, 20)
    a = random.randint(0, pow(10, n1))
    b = random.randint(0, pow(10, n2))
    return f"{a}+{b}={a}+{b}={a+b}"


@registered()
def doublehistogram():
    vocab = string.ascii_lowercase + string.ascii_uppercase
    n = random.randint(10, 80)
    letters = ''.join(random.choices(vocab, k=n))
    h1 = [letters.count(t) for t in letters]
    res = [h1.count(c) // c for c in h1]
    return letters + "::" + str(res)


@registered()
def sort():
    n = random.randint(10, 60)
    # 2*60 +~10 = 130 < 200 (normally my max length is 200)
    vocab = string.ascii_lowercase + string.ascii_uppercase + string.digits
    letters = ''.join(random.choices(vocab, k=n))
    return letters + " :: " + ''.join(sorted(letters))


@registered()
def copy():
    n = random.randint(10, 60)
    # 2*60 +~10 = 130 < 200 (normally my max length is 200)
    vocab = string.ascii_lowercase + string.ascii_uppercase + string.digits
    letters = ''.join(random.choices(vocab, k=n))
    return letters + " :: " + letters


@registered()
def numbersort():
    # going to sort 3-digit numbers
    n = random.randint(10, 30)
    # (30*4)*3 + ~10 = 370  -> need to be running at length up to 370
    numbers = [random.randint(0, 999) for _ in range(n)]

    def tostr(lst):
        return f"[{','.join('{:03d}'.format(v) for v in lst)}]"
    return tostr(numbers) + "--" + tostr(numbers) + "--" +\
        tostr(sorted(numbers))


############################################
# functions for checking model generations #
############################################


def check_long_addition(s):
    # assumes no X scratchpad, will have to edit to allow those
    main_s = [s[i] for i in range(len(s)) if i % 2 == 0]
    spaces = [s[i] for i in range(len(s)) if i % 2 == 1]
    if ' ' in main_s:
        return False, "poor form: spaces in even spaces"
    if not (list(set(spaces)) == [' ']):
        return False, "poor form: non-spaces in odd spaces"
    s = ''.join(main_s)
    if not s.count("=") == 2:
        return False, f"poor form: {s.count('=')} ='s"
    bits = s.split("=")
    if not bits[0].count("+") == 1:
        return False, f"poor form: {bits[0].count('+')} +'s in statement"
    if not bits[1].count("+") == 1:
        return False, f"poor form: {bits[1].count('+')} +'s in repeat"
    if not bits[2].count("+") == 0:
        return False, f"poor form: {bits[2].count('+')} +'s in answer"
    a, b = bits[0].split("+")
    a2, b2 = bits[1].split("+")
    if a[0] == '0':
        return False, "poor form: leading 0 in first number"
    if b[0] == '0':
        return False, "poor form: leading 0 in second number"
    if not (a == a2):
        return False, "poor form: first number not repeated correctly"
    if not (b == b2):
        return False, "poor form: second number not repeated correctly"
    a, b = int(a), int(b)
    c = bits[2]
    if c[0] == '0':
        return False, "poor form: leading 0 in result"
    if a + b == int(c):
        return True, ""
    else:
        return False, f"wrong result. expected {a + b}, got {int(c)}"
