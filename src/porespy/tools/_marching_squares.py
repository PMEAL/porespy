import numpy as np
from numba import njit


__all__ = [
    'jit_marching_squares_perimeter_and_area',
]


TEMPLATE_PERIMETER = (
    0,
    1,
    1,
    2,
    1,
    3,
    4,
    1,
    1,
    4,
    3,
    1,
    2,
    1,
    1,
    0,
)

TEMPLATE_AREA = (
    0.,
    1/8,
    1/8,
    1/2,
    1/8,
    3/4,
    1/2,
    7/8,
    1/8,
    1/2,
    3/4,
    7/8,
    1/2,
    7/8,
    7/8,
    1.,
)

TEMPLATE_REPETITION = (
    0.,
    1.,
    1.,
    1/2,
    1.,
    1/2,
    1/2,
    1/3,
    1.,
    1/2,
    1/2,
    1/3,
    1/2,
    1/3,
    1/3,
    1/4,
)


@njit
def jit_marching_squares_perimeter_and_area(
    img,
    target_label=1,
    spacing=(1., 1.),
    overlap=False
):
    w, h = spacing
    complete_area = w * h
    perimeters = [
        0,
        np.sqrt(w**2 + h**2)/2,
        w,
        np.sqrt(w**2 + h**2),
        h,
    ]

    total_perimeter = 0
    total_area = 0
    max_x, max_y = img.shape
    for x in range(max_x - 1):
        for y in range(max_y - 1):
            template = 0
            if img[x, y] == target_label:
                template += 1
            if img[x + 1, y] == target_label:
                template += 2
            if img[x, y + 1] == target_label:
                template += 4
            if img[x + 1, y + 1] == target_label:
                template += 8
            if overlap:
                total_perimeter += perimeters[TEMPLATE_PERIMETER[template]] \
                    * TEMPLATE_REPETITION[template]
                total_area += complete_area * TEMPLATE_AREA[template] \
                    * TEMPLATE_REPETITION[template]
            else:
                total_perimeter += perimeters[TEMPLATE_PERIMETER[template]]
                total_area += complete_area * TEMPLATE_AREA[template]

    return total_perimeter/2, total_area
