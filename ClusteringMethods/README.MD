# Изменено:
  - Добалена более обширная документация для всех функций


# Документация по методам кластеризации

## Авторы
- **Mineev S. A.** [mineeff20@yandex.ru]
- **Meshkova O. V.** [oxn.lar5@yandex.ru]
- **Kozlov A. I.** [alex_kozlov15@mail.ru]

Ниже представлены методы кластеризации, используемые в проекте версии v1.1. Включены как подробные описания параметров, так и практические примеры использования, чтобы сделать работу с кодом более понятной и легкой для дальнейшего улучшения.

### 1. BIRCH из библиотеки scikit-learn

**Установка:**
```bash
python -m pip install scikit-learn
```
**Импорт:**
```python
from sklearn.cluster import Birch
```

**Описание метода:**
Метод BIRCH (Balanced Iterative Reducing and Clustering using Hierarchies) из библиотеки scikit-learn используется для кластеризации больших наборов данных, работая с ограниченными вычислительными ресурсами.

**Сигнатура:**
```python
class sklearn.cluster.Birch(
    *, threshold=0.5, branching_factor=50, n_clusters=3, compute_labels=True, copy=True
)
```

**Параметры:**
- **threshold** (*float*, по умолчанию 0.5): Радиус подкластера, полученный путем слияния нового образца и ближайшего подкластерного центра. Если радиус превышает значение порога, создается новый подкластер. Установка низкого значения способствует более дробному делению кластеров.

- **branching_factor** (*int*, по умолчанию 50): Максимальное количество подкластеров (CF) в каждом узле. Если количество подкластеров превышает значение этого параметра, узел разделяется на два, что приводит к перераспределению подкластеров между ними.

- **n_clusters** (*int, экземпляр модели sklearn.cluster или None*, по умолчанию 3): Количество кластеров на завершающем этапе. Листовые подгруппы рассматриваются как новые точки данных.

- **compute_labels** (*bool*, по умолчанию True): Определяет, вычислять ли метки для каждого объекта данных.

- **copy** (*bool*, по умолчанию True): Определяет, следует ли создавать копию входных данных. Если False, исходные данные будут изменены.


**Источник:** [Документация scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.Birch.html)

---

### 2. BIRCH из библиотеки pyclustering

**Установка:**
```bash
python -m pip install pyclustering
```
**Импорт:**
```python
from pyclustering.cluster.birch import birch
from pyclustering.container.cftree import measurement_type
```

**Описание метода:**
Метод BIRCH из библиотеки pyclustering используется для создания CF-дерева и дальнейшей кластеризации данных с использованием заданных параметров.

**Сигнатура:**
```python
birch(
    data, number_clusters=3, branching_factor=50, max_node_entries=3,
    diameter=0.5, type_measurement=measurement_type.CENTROID_EUCLIDEAN_DISTANCE,
    entry_size_limit=500, diameter_multiplier=1.5, ccore=True
)
```

**Параметры:**
- **data** (*list*): Входные данные, представленные в виде списка точек, каждая из которых является списком координат.

- **number_clusters** (*int*, по умолчанию 3): Количество кластеров, которые нужно выделить.

- **branching_factor** (*int*, по умолчанию 50): Максимальное количество подгрупп (CF) в каждом нелистовом узле CF-дерева.

- **max_node_entries** (*int*, по умолчанию 200): Максимальное количество элементов, которые могут содержаться в каждом листовом узле CF-дерева.

- **diameter** (*float*, по умолчанию 0.5): Диаметр CF-входного узла, используемый для построения дерева. Может увеличиваться, если превышен лимит элементов в узле.

- **type_measurement** (*measurement_type*, по умолчанию `measurement_type.CENTROID_EUCLIDEAN_DISTANCE`): Тип метрики, используемой для вычисления расстояний.

- **entry_size_limit** (*int*, по умолчанию 500): Максимальное количество записей, которое может быть в дереве. Если лимит превышен, диаметр увеличивается, и дерево перестраивается.

- **diameter_multiplier** (*float*, по умолчанию 1.5): Множитель для увеличения диаметра при превышении лимита записей.

- **ccore** (*bool*, по умолчанию True): Если True, то для обработки используется часть библиотеки C++.


**Источник:** [Документация pyclustering](https://pyclustering.github.io/docs/0.10.1/html/d6/d00/classpyclustering_1_1cluster_1_1birch_1_1birch.html)

---

### 3. CURE из библиотеки pyclustering

**Установка:**
```bash
python -m pip install pyclustering
```
**Импорт:**
```python
from pyclustering.cluster.cure import cure
```

**Описание метода:**
Метод CURE (Clustering Using Representatives) используется для уменьшения размеров кластеров путем сжатия выбранных репрезентативных точек.

**Сигнатура:**
```python
cure(data, number_clusters=3, number_represent_points=5, compression=0.5, ccore=True)
```

**Параметры:**
- **data** (*list*): Входные данные, представленные в виде списка точек (объектов), где каждая точка - это список координат.

- **number_clusters** (*int*, по умолчанию 3): Количество кластеров, которые должны быть выделены.

- **number_represent_points** (*int*, по умолчанию 5): Количество репрезентативных точек для каждого кластера.

- **compression** (*float*, по умолчанию 0.5): Коэффициент, определяющий уровень сжатия точек представления к центроиду нового кластера после объединения.

- **ccore** (*bool*, по умолчанию True): Если True, то для обработки используется часть библиотеки C++.


**Источник:** [Документация pyclustering](https://pyclustering.github.io/docs/0.10.0/html/dc/d6d/classpyclustering_1_1cluster_1_1cure_1_1cure.html)

---

### 4. ROCK из библиотеки pyclustering

**Установка:**
```bash
python -m pip install pyclustering
```
**Импорт:**
```python
from pyclustering.cluster.rock import rock
```

**Описание метода:**
Метод ROCK (Robust Clustering using Links) используется для кластеризации категориальных данных и основан на количестве "связей" между точками.

**Сигнатура:**
```python
rock(data, eps, number_clusters=3, threshold=0.5, ccore=True)
```

**Параметры:**
- **data** (*list*): Входные данные, представленные в виде списка точек, где каждая точка - это список координат.

- **eps** (*float*): Радиус связности. Точки являются соседними, если расстояние между ними меньше заданного радиуса.

- **number_clusters** (*int*, по умолчанию 3): Количество кластеров, которые должны быть выделены.

- **threshold** (*float*, по умолчанию 0.5): Порог нормализации, влияющий на выбор кластеров для объединения.

- **ccore** (*bool*, по умолчанию True): Определяет, должен ли CCORE (библиотека C ++ для пикластеринга) использоваться вместо кода Python или нет.


**Источник:** [Документация pyclustering](https://pyclustering.github.io/docs/0.8.2/html/d8/dde/classpyclustering_1_1cluster_1_1rock_1_1rock.html)

---



