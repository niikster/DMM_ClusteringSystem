import time

import numpy as np
from numpy import linspace, cos, sin, pi
from numpy.random import default_rng
from matplotlib import pyplot as plt

from sklearn.datasets import make_moons
# Скрипт предназначен для генерации данных с помощью метода make_moons из библиотеки scikit-learn.
# make_moons используется для создания набора точек, представляющих собой две пересекающиеся полулуны, что полезно для тестирования алгоритмов кластеризации и классификации.

if __name__ == '__main__':
    tic = time.process_time()
    Data, label = make_moons()
    toc = time.process_time()
    print(f"Вычисление заняло {toc - tic:0.4f} секунд")

# Выходные данные:
# - Data: массив точек, представляющий координаты точек полулун.
# - label: метки классов для каждого элемента массива Data.    