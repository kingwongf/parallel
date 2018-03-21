import multiprocessing
import numpy as np
from itertools import repeat

def f1(x):
    """worker function"""
    a = np.mean(x)*np.random.rand(100)
    b = x*np.random.randn(10)
    print(a)
    return a,b
def f2(x, y):
    x =x
    for i in range(1000):
        pool = multiprocessing.Pool()
        a,b= pool.map(f1, repeat(x))

    out1 = np.mean(a)
    out2 = np.mean(b)

if __name__ == '__main__':
    x = [1,2,3,4,5]
    for i in range(10000):
        p = multiprocessing.Process(target=worker)
        jobs.append(p)
        p.start()