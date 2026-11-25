"""Compare morphology methods; correctness and JIT warm-up precede timing.

Run from the repository root with:
    NUMBA_NUM_THREADS=4 .venv/bin/python scripts/benchmark_sphere_morphology.py
"""
from statistics import median
from time import perf_counter

import numpy as np

import porespy as ps


def main():
    print('shape,porosity,radius,operation,method,median_seconds')
    for shape in ((256, 256), (64, 64, 64)):
        for porosity in (0.2, 0.8):
            im = ps.generators.blobs(shape=shape, porosity=porosity, seed=5)
            for radius in (2, 5, 10):
                for operation in ('erode', 'dilate'):
                    func = getattr(ps.filters, operation)
                    expected = func(im, radius, method='dt')
                    for method in ('dt', 'conv', 'brute_force'):
                        actual = func(im, radius, method=method)
                        np.testing.assert_array_equal(actual, expected)
                        times = []
                        for _ in range(3):
                            start = perf_counter()
                            func(im, radius, method=method)
                            times.append(perf_counter() - start)
                        label = 'x'.join(map(str, shape))
                        print(f'{label},{porosity},{radius},{operation},'
                              f'{method},{median(times):.6f}', flush=True)


if __name__ == '__main__':
    main()
