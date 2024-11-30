import time

from matplotlib import pyplot as plt

import numpy as np
from numpy import linspace, cos, sin, pi
from numpy.random import default_rng

from sklearn.datasets import make_circle
# Скрипт предназначен для генерации данных с помощью метода make_circle из библиотеки scikit-learn.
# make_circle используется для создания набора точек, образующих круг, что полезно для тестирования алгоритмов кластеризации и классификации.

if __name__ == '__main__':
    tic = time.process_time()
    # Генерация выборки, представляющей собой точки, образующие круг
    Data, label = make_circle()
    toc = time.process_time()
    print(f"Вычисление заняло {toc - tic:0.4f} секунд")

# Выходные данные:
# - Data: массив точек, представляющий координаты точек на окружности.
# - label: метки классов для каждого элемента массива Data.