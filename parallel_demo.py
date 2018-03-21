from functools import partial
from itertools import repeat
from multiprocessing import Pool, freeze_support
import numpy as np

hold1 = [9,9,9,9,9]

# def f1(x):
#     """worker function"""
#     a = np.mean(x)*np.random.rand(100)
#     b = np.mean(x)*np.random.randn(10)
#     print(a)
#     return a,b
# if __name__ == '__main__':
#     x = [1, 2, 3, 4, 5]
#     out1list = []
#     out2list = []
#     for i in range(1000):
#         pool = multiprocessing.Pool()
#         a,b= pool.map(f1, repeat(x))
#         out1list.append(a)
#         out2list.append(b)
#     print(out1list, out2list)

def funSquare(num):
    return num ** 2

def func(a, b):
    np.random.seed()
    # x = np.random.randn(1)
    x = 1
    return np.multiply(hold1,b)

def main():
    a_args = range(1000)
    second_arg = [1,3,4,5,6]
    with Pool() as pool:
        # for i in range(1000):
        #     M = pool.starmap(func, zip(a_args, repeat(second_arg)))
        #     print(M)
        M = pool.starmap(func, zip(a_args, repeat(second_arg)))
        print(len(M))

if __name__=="__main__":
    freeze_support()
    main()

