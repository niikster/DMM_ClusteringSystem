# Authors:
#       Mineev S. A. [mineeff20@yandex.ru]
#       Meshkova O. V. [oxn.lar5@yandex.ru]

from __future__ import annotations
from abc import ABC, abstractmethod

from dataclasses import dataclass
from enum import Enum, auto
import re

import numpy as np                      # pip install numpy
from pyclustering.cluster.birch import birch
from pyclustering.cluster.cure import cure
from pyclustering.cluster.rock import rock
from pyclustering.container.cftree import measurement_type
from sklearn.cluster import Birch       # pip install sklearn-learn

from typing import Dict, List


@dataclass
class StrategyParam:
    ui_name: str
    param_type: StrategyParamType
    switches: List[str]
    description: str
    default_value: int | float | str | bool


class StrategyRunConfig:
    _values: Dict[str, int | float | str | bool] = dict()

    def __init__(self, params: Dict[str, StrategyParam]):
        self._params = params

    @property
    def params(self):
        return self._params

    def __getitem__(self, key: str):
        if key not in self._params:
            raise IndexError(f"config does not have field {key}")

        if key in self._values:
            return self._values[key]
        
        return self._params[key].default_value

    def __setitem__(self, key: str, val: int | float | str | bool):
        if key not in self._params:
            raise IndexError(f"config does not have field {key}")

        paramType = self._params[key]

        if not StrategyParamType.checkValIsExpected(paramType.param_type, val, switches=paramType.switches):
            raise ValueError(f"config expected for field {key} type {paramType.param_type.name} but got {type(val)} ({val})")

        self._values[key] = val


class Context:
    """
    Контекст определяет интерфейс, представляющий интерес для клиентов.
    """

    def __init__(self, strategy: Strategy) -> None:
        """
        Обычно Контекст принимает стратегию через конструктор, а также
        предоставляет сеттер для её изменения во время выполнения.
        """

        self._strategy = strategy

    @property
    def strategy(self) -> Strategy:
        """
        Контекст хранит ссылку на один из объектов Стратегии. Контекст не знает
        конкретного класса стратегии. Он должен работать со всеми стратегиями
        через интерфейс Стратегии.
        """
        return self._strategy

    @strategy.setter
    def strategy(self, strategy: Strategy) -> None:
        """
        Обычно Контекст позволяет заменить объект Стратегии во время выполнения.
        """
        self._strategy = strategy

    def do_some_clustering_image(self, pixels: np.ndarray, params: StrategyRunConfig, i) -> np.ndarray:
        """
        Вместо того, чтобы самостоятельно реализовывать множественные версии
        алгоритма, Контекст делегирует некоторую работу объекту Стратегии.
        """
        if i > 0:
            # Извлечение координат и цветов всех пикселей в формате HSV
            coords_and_colors = [
                (x, y, pixels[y, x])
                for y in range(pixels.shape[0])
                for x in range(pixels.shape[1])
            ]
            # Преобразование координат и цветов в двумерный массив для кластеризации
            # X = np.array([[h, s, v] for (_, _, (h, s, v)) in coords_and_colors])
            X = [[float(h), float(s), float(v)]
                 for (_, _, (h, s, v)) in coords_and_colors]
            # ------------------------------------------------------------------------------------- #
            labels = self._strategy.clastering_image(X, params)
            # ------------------------------------------------------------------------------------- #
            # Визуализация результата кластеризации
            colors = [
                (255, 0, 0),
                (0, 255, 0),
                (0, 0, 255),
                (255, 255, 0),
                (255, 0, 255),
                (0, 255, 255),
            ]
            clustered_image = np.zeros_like(pixels)
            for index, (x, y, _) in enumerate(coords_and_colors):
                clustered_image[y, x] = colors[labels[index] % len(colors)]
        else:
            # None(used Rashape)
            labels = self._strategy.clastering_image(pixels.tolist(), params)
        return labels

    def do_some_clustering_points(self, data, params: StrategyRunConfig) -> np.ndarray:
        fitchs = np.array(data, dtype=float)
        points = fitchs.transpose()
        if isinstance(self._strategy, ConcreteStrategyBIRCH_from_SKLEARN_LEARN):
            labels = self._strategy.clastering_points(points, params).tolist()
        else:
            labels = self._strategy.clastering_points(points, params)
        return labels


class StrategyParamType(Enum):
    Number = auto()
    UNumber = auto()
    Floating = auto()
    UFloating = auto()
    Switch = auto()
    Bool = auto()

    @staticmethod
    def checkValIsExpected(enumValue: StrategyParamType, value: int | float | bool | str, switches: List[str] = []) -> bool:
        match enumValue:
            case StrategyParamType.Number:
                return isinstance(value, int)
            case StrategyParamType.UNumber:
                return isinstance(value, int) and value >= 0
            case StrategyParamType.Floating:
                return isinstance(value, float)
            case StrategyParamType.UFloating:
                return isinstance(value, float) and value >= 0
            case StrategyParamType.Switch:
                return isinstance(value, str) and value in switches
            case StrategyParamType.Bool:
                return isinstance(value, bool)

        return False


class Strategy(ABC):
    """
    Интерфейс Стратегии объявляет операции, общие для всех поддерживаемых версий
    некоторого алгоритма.

    Контекст использует этот интерфейс для вызова алгоритма, определённого
    Конкретными Стратегиями.
    """

    @classmethod
    def params(cls):
        """Возвращает набор параметров метода кластеризации

        Returns:
            Dict[str, StrategyParam]: Набор параметров метода кластеризации
        """
        return cls._params

    @classmethod
    @abstractmethod
    def _setupParams(cls):
        """Инициализирует список параметров настроек стратегии

        Здесь наследники стратегии должны при помощи _addParam добавить все поддерживаемые стратегией опции.
        """
        pass

    @classmethod
    def _addParam(cls, id: str, name: str, param_type: StrategyParamType, descr: str, default_value: int | float | str | bool, switches: List[str] = list()):
        """Добавляет параметр настройки стратегии

        Args:
            id (str): Идентификатор параметра, передаваемого объекта настройки стратегии (см. StrategyRunConfig)
            name (str): Человеко-читаемое имя параметра
            param_type (StrategyParamType): Тип значения параметра
            descr (str): Полное описание параметра
            default_value (int | float | str | bool): Значение по умолчанию. Если не задано, то будет использовано 0
                             для численных и первое значение из параметра с выбором из опций.
            switches (List[str], optional): Возможные значения параметра с типа "Выбор из нескольких". По умолчанию список пустой.

        Raises:
            ValueError: Возникает при попытке установить возможный параметр типа StrategyParamType.Switch без единого возможного значения
        """

        if param_type == StrategyParamType.Switch:
            if len(switches) <= 0:
                raise ValueError("must be at least one switch")
            if default_value is None:
                default_value = switches[0]

        lowerId = id.lower().strip().replace(" ", "").replace("-", "_")

        if not hasattr(cls, "_params"):
            cls._params = dict()

        cls._params[lowerId] = StrategyParam(
            name, param_type, switches, re.sub('\\s{2,}', ' ', descr.strip()), default_value)

    @abstractmethod
    def clastering_image(self, pixels: np.ndarray, params: StrategyRunConfig) -> np.ndarray:
        pass

    @abstractmethod
    def clastering_points(self, points: np.ndarray, params: StrategyRunConfig) -> np.ndarray:
        pass

    def clusters_to_labels(self, clusters) -> List:
        """
        @brief Метод преобразования cluster_index in labels для birch из sklearn-learn.
        """
        size = sum(map(lambda x: len(x), clusters))
        labels = [0] * size
        for cluster_index in range(len(clusters)):
            for i in clusters[cluster_index]:
                labels[i] = cluster_index

        return labels


@dataclass
class StrategyDescription:
    """Описание стратегии кластеризации
    """

    """Класс стратегии
    """
    strategyType: type 

    """Название стратегии
    """
    name: str

    """Краткое описание стратегии
    """
    description: str


class StrategiesManager:
    _strategies: Dict[str, StrategyDescription] = dict()

    @classmethod
    def strategies(cls):
        return cls._strategies

    @classmethod
    def registerStrategy(cls, id: str, ui_name: str, description=""):
        """Декоратор для регистрации стратегии

        Args:
            id (str): Идентификатор стратегии. При помощи него в дальнейшем можно
                      создать новый экземпляр стратегии
            ui_name (str): Человеко-читаемое название стратегии
            description (str, optional): Описание стратегии. По умолчанию не задано.

        Raises:
            TypeError: Возникает при попытке добавить стратегию по уже занятому ID.
        """

        lowerId = id.lower().strip().replace(" ", "").replace("-", "_")

        def _innerDecor(strategyClass: type):
            if lowerId in cls._strategies:
                raise TypeError(
                    f"Id {lowerId} is already in use. Tried to register class named {strategyClass.__name__}")

            strategyClass._setupParams()
            cls._strategies[lowerId] = StrategyDescription(
                strategyClass, ui_name, description)
            return strategyClass
        return _innerDecor

    @classmethod
    def createStrategyById(cls, id) -> Strategy | None:
        if id in cls._strategies:
            return cls._strategies[id].strategyType()

        return None

    @classmethod
    def strategiesCount(cls) -> int:
        return len(cls._strategies)

    @classmethod
    def getStrategyRunConfigById(cls, id: str) -> StrategyRunConfig | None:
        if id in cls._strategies:
            return StrategyRunConfig(cls._strategies[id].strategyType.params())
        
        return None


"""
Конкретные Стратегии реализуют алгоритм, следуя базовому интерфейсу Стратегии.
Этот интерфейс делает их взаимозаменяемыми в Контексте.
"""


@StrategiesManager.registerStrategy("birch_sk", "BIRCH (SKLearn)")
class ConcreteStrategyBIRCH_from_SKLEARN_LEARN(Strategy):

    @classmethod
    def _setupParams(cls):
        cls._addParam("n_clusters", "Количество кластеров", StrategyParamType.UNumber, """
            (Number of clusters) Количество кластеров после заключительного этапа кластеризации,
            на котором подкластеры из листьев рассматриваются как новые образцы""", 3)

        cls._addParam("branching_factor", "Коэффициент ветвления", StrategyParamType.UNumber,
            """(Branching factor) Максимальное число CF подкластеров в каждом узле""",
            50)

        cls._addParam("threshold", "Пороговая величина", StrategyParamType.UFloating, """
            (Threshold) Радиус подкластера, полученного путем слияния новой выборки и ближайшего
            подкластера, должен быть меньше порогового значения""", 0.5)

        cls._addParam("compute_labels", "Вычисление меток", StrategyParamType.Bool, """
            (Compute labels) Следует ли вычислять метки для каждого соответствия или нет""",
            True)

        cls._addParam("copy", "Copy data", StrategyParamType.Bool, """
            (Copy data) Следует ли делать копию предоставленных данных или нет""", True)

    '''
        @brief метод кластеризации изображений с использованием birch из sklearn-learn.

        @param[in] self (obj): The current object.
        @param[in] pixels (list): the image represented by pixels.
        @param[in] params (list): the parameters for clustering.

        @return (list) The label for everyone pixel.
    '''

    def clastering_image(self, pixels: np.ndarray, params: StrategyRunConfig) -> np.ndarray:
        # Создание и обучение объекта BIRCH
        birch1 = Birch(n_clusters=int(params["n_clusters"]), branching_factor=int(params["branching_factor"]),
                       threshold=float(params["threshold"]), compute_labels=bool(params["compute_labels"]),
                       copy=bool(params["copy"]))
        birch1.fit(pixels)

        # Получение меток кластеров для каждой точки
        return birch1.predict(pixels)

    '''
        @brief метод кластеризации точек с использованием birch из sklearn-learn.

        @param[in] self (obj): The current object.
    	@param[in] points (list): the image represented by points.

    	@return (list) The label for everyone point.
    '''

    def clastering_points(self, points: np.ndarray, params: StrategyRunConfig) -> np.ndarray:
        # Создание и обучение объекта BIRCH
        birch1 = Birch(n_clusters=int(params["n_clusters"]), branching_factor=int(params["branching_factor"]),
                       threshold=float(params["threshold"]), compute_labels=bool(params["compute_labels"]),
                       copy=bool(params["copy"]))
        birch1.fit(points)
        # Получение меток кластеров для каждой точки
        if params["compute_labels"]:
            labels = birch1.labels_
        else:
            labels = birch1.predict(points)
        return labels


@StrategiesManager.registerStrategy("birch_pyc", "BIRCH (PyClustering)")
class ConcreteStrategyBIRCH_from_PYCLUSTERING(Strategy):

    @classmethod
    def _setupParams(cls):
        cls._addParam("n_clusters", "Количество кластеров", StrategyParamType.UNumber, """
            (Number of clusters) Количество кластеров после заключительного этапа кластеризации,
            на котором подкластеры из листьев рассматриваются как новые образцы""", 3)
        cls._addParam("branching_factor", "Коэффициент ветвления", StrategyParamType.UNumber,
            "(Branching factor) Максимальное число CF подкластеров в каждом узле", 50)
        cls._addParam("max_node_entries", "Максимальное количество записей в узлах",
            StrategyParamType.UNumber, """(Maximum number of node entries) Максимальное
            количество записей, которые могут содержаться в каждом конечном узле CF-дерева""",
            200)
        cls._addParam("diameter", "Диаметр CF-записи", StrategyParamType.UFloating, """
            (CF-entry diameter) Диаметр CF-записи, который используется для конструирования
            CF-дерева""", 0.5)
        cls._addParam("type_measurement", "Тип измерения", StrategyParamType.Switch,
            """(Type measurement) Тип измерения, используемого для расчета показателей расстояния""",
            "Euclidean",
            switches=["Euclidean", "Manhattan", "Inter", "Intra", "Increase"])
        cls._addParam("entry_size_limit", "Предельный размер записей", StrategyParamType.UNumber,
            """(Entry size limit) Максимальное количество записей, которое может быть сохранено в
            CF-дереве, если оно превышено""", 500)
        cls._addParam("diameter_multiplier", "Множитель диаметра", StrategyParamType.Floating,
            """Множитель, который используется для увеличения диаметра при превышении
            "Предельного размера записей".""", 1.5)
        cls._addParam("ccore", "Использовать C++", StrategyParamType.Bool, """
            Если истинно, тогда используется C++ часть библиотеки для обработки""", True)

    """
        @brief метод кластеризации пикселей с использованием birch из pyclustering.

        @param[in] self (obj): The current object.
    	@param[in] points (list): the image represented by pixels.

    	@return (list) The label for everyone point.
    """

    def clastering_image(self, pixels: np.ndarray, params: StrategyRunConfig) -> np.ndarray:
        typeMeasurement = self.TYPE(params["type_measurement"])
        instance = birch(data=pixels, number_clusters=int(params["n_clusters"]), branching_factor=int(params["branching_factor"]),
                         max_node_entries=int(params["max_node_entries"]), diameter=float(params["diameter"]), type_measurement=typeMeasurement,
                         entry_size_limit=int(params["entry_size_limit"]), diameter_multiplier=float(params["diameter_multiplier"]), ccore=bool(params["ccore"]))
        instance.process()
        return np.array(self.clusters_to_labels(instance.get_clusters()))

    '''
        @brief метод кластеризации точек с использованием birch из pyclustering.

        @param[in] self (obj): The current object.
    	@param[in] points (list): the image represented by points.
    	@param[in] params (list): the parameters for clustering.

    	@return (list) The label for everyone point.
    '''

    def clastering_points(self, points, params):
        typeMeasurement = self.TYPE(params["type_measurement"])
        instance = birch(data=points.tolist(), number_clusters=int(params["n_clusters"]), branching_factor=int(params["branching_factor"]),
                         max_node_entries=int(params["max_node_entries"]), diameter=float(params["diameter"]), type_measurement=typeMeasurement,
                         entry_size_limit=int(params["entry_size_limit"]), diameter_multiplier=float(params["diameter_multiplier"]), ccore=bool(params["ccore"]))
        instance.process()
        return np.array(self.clusters_to_labels(instance.get_clusters()))

    '''
        @brief метод, определяющий тип метрики для birch из pyclustering.

        @param[in] self (obj): The current object.
    	@param[in] param (list): the index metric from mainwindow.py.

    	@return (measurement_type: int) The type metric.
    '''

    def TYPE(self, param):
        match param:
            case 'Euclidean':
                type = measurement_type.CENTROID_EUCLIDEAN_DISTANCE
            case 'Manhattan':
                type = measurement_type.CENTROID_MANHATTAN_DISTANCE
            case 'Inter':
                type = measurement_type.AVERAGE_INTER_CLUSTER_DISTANCE
            case 'Intra':
                type = measurement_type.AVERAGE_INTRA_CLUSTER_DISTANCE
            case 'Increase':
                type = measurement_type.VARIANCE_INCREASE_DISTANCE
            case _:
                type = measurement_type.CENTROID_EUCLIDEAN_DISTANCE
        return type


@StrategiesManager.registerStrategy("cure", "CURE")
class ConcreteStrategyCURE(Strategy):

    @classmethod
    def _setupParams(cls):
        cls._addParam("n_clusters", "Количество выделенных кластеров", StrategyParamType.UNumber,
            "(Number of allocated clusters) Количество кластеров для выделения", 50)
        cls._addParam("number_represent_points", "Количество репрезентативных точек", StrategyParamType.UNumber,
            "(Number of representative points) Количество репрезентативных точек для каждого кластера", 5)
        cls._addParam("compression", "Сжатие", StrategyParamType.Floating,
            """(Compression) Коэффициент определяет уровень уменьшения количества точек представления
            по отношению к среднему значению вновь созданного кластера после объединения на каждом шаге.
            Обычно находится в диапазоне от 0 до 1""", 0.5)
        cls._addParam("ccore", "Использовать C++", StrategyParamType.Bool, """
            Если истинно, тогда используется C++ часть библиотеки для обработки""", True)

    '''
        @brief метод кластеризации пикселей с использованием cure из pyclustering.

        @param[in] self (obj): The current object.
        @param[in] pixels (list): the image represented by points.

        @return (list) The label for every point.
    '''

    def clastering_image(self, pixels: np.ndarray, params: StrategyRunConfig) -> np.ndarray:
        instance = cure(data=pixels, number_cluster=int(params["n_clusters"]),
                        number_represent_points=int(params["number_represent_points"]),
                        compression=float(params["compression"]), ccore=bool(params["ccore"]))
        instance.process()
        return np.array(self.clusters_to_labels(instance.get_clusters()))

    '''
        @brief метод кластеризации точек с использованием cure из pyclustering.

        @param[in] self (obj): The current object.
    	@param[in] points (list): the image represented by points.
    	@param[in] params (list): the parameters for clustering.

    	@return (list) The label for everyone point.
    '''

    def clastering_points(self, points: np.ndarray, params: StrategyRunConfig) -> np.ndarray:
        instance = cure(data=points.tolist(), number_cluster=int(params["n_clusters"]),
                        number_represent_points=int(params["number_represent_points"]),
                        compression=float(params["compression"]), ccore=bool(params["ccore"]))
        instance.process()
        return np.array(self.clusters_to_labels(instance.get_clusters()))


@StrategiesManager.registerStrategy("rock", "ROCK")
class ConcreteStrategyROCK(Strategy):

    @classmethod
    def _setupParams(cls):
        cls._addParam("eps", "Радиус соединения (порог сходства)", StrategyParamType.Floating,
            """(Connectivity radius (similarity threshold)) Радиус соединения (порог сходства),
            точки являются соседями, если дистанция между ними ниже радиуса соединения""", 2.0)
        cls._addParam("n_clusters", "Количество кластеров", StrategyParamType.UNumber,
            """(Number of clusters) Определяет число кластеров, которое должно быть выделено
            из входного набора данных""", 3)
        cls._addParam("threshold", "Пороговая величина", StrategyParamType.Floating, """
            (Threshold) Значение, определяющее степень нормализации, которая влияет на выбор
            кластеров для объединения во время обработки.""", 0.5)
        cls._addParam("ccore", "Использовать C++", StrategyParamType.Bool, """
            Если истинно, тогда используется C++ часть библиотеки для обработки""", True)

    '''
        @brief метод кластеризации пикселей с использованием rock из pyclustering.

        @param[in] self (obj): The current object.
        @param[in] pixels (list): the image represented by pixels.

        @return (list) The label for every point.
    '''

    def clastering_image(self, pixels: np.ndarray, params: StrategyRunConfig) -> np.ndarray:
        instance = rock(data=pixels, eps=params["eps"], number_clusters=int(params["n_clusters"]),
                        threshold=float(params["threshold"]), ccore=bool(params["ccore"]))
        instance.process()
        return np.array(self.clusters_to_labels(instance.get_clusters()))

    '''
        @brief метод кластеризации точек с использованием rock из pyclustering.

        @param[in] self (obj): The current object.
    	@param[in] points (list): the image represented by points.
    	@param[in] params (list): the parameters for clustering.

    	@return (list) The label for every point.
    '''

    def clastering_points(self, points: np.ndarray, params: StrategyRunConfig):
        instance = rock(data=points.tolist(), eps=params["eps"], number_clusters=int(params["n_clusters"]),
                        threshold=float(params["threshold"]), ccore=bool(params["ccore"]))
        instance.process()
        x = instance.get_clusters()
        return np.array(self.clusters_to_labels(x))
