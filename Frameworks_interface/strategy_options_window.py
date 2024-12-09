import sys
import ctypes

from PySide6.QtCore import Signal, Slot, qDebug
from PySide6.QtGui import Qt
from PySide6.QtWidgets import QCheckBox, QComboBox, QDoubleSpinBox, QHBoxLayout, QLabel, QSpinBox, QWidget

from ClusteringMethods.ClasteringAlgorithms import StrategiesManager, StrategyParamType, StrategyRunConfig

from .ui_strategy_options_dialog import Ui_StrategyOptionsDialog

from typing import Dict, Optional, Tuple, Union


class StrategyOptionsDialog(QWidget):
    # Signals
    optionsApplied = Signal(StrategyRunConfig)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent, f=Qt.WindowType.Dialog)

        self.__ui = Ui_StrategyOptionsDialog()
        self.__ui.setupUi(self)

        self.setWindowModality(Qt.WindowModality.WindowModal)

        # Обработка сигналов
        self.__ui.okButton.clicked.connect(self.__saveParams)

        self.__stratId = None
        self.__runConfig = None
        self.__fieldWidgets: Dict[str, Tuple[QWidget, Union[QSpinBox, QDoubleSpinBox, QCheckBox, QComboBox]]] = dict()
        
    def setRunConfig(self, startId: str, stratRunConfig: StrategyRunConfig) -> None:
        self.__stratId = startId
        self.__runConfig = stratRunConfig

        self.setWindowTitle(f"Настройки для {StrategiesManager.strategies()[self.__stratId].name}")

        self.__recreateParamFields()

    def __recreateParamFields(self) -> None:
        if self.__stratId is None or self.__runConfig is None:
            qDebug("StrategyOptionsWindow tried to recreate param fields with non-existent values")
            return

        # Очищаем все поля опций, если были до этого
        while self.__ui.optionsLayout.count() != 0:
            item = self.__ui.optionsLayout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
            
            self.__ui.optionsLayout.removeItem(item)
        
        for _, (fieldWidget, valueWidget) in self.__fieldWidgets.items():
            fieldWidget.deleteLater()
            valueWidget.deleteLater()
        self.__fieldWidgets.clear()
        
        # Создаём на основе параметров стратегии
        for paramId, param in self.__runConfig.params.items():
            widget = QWidget()
            widget.setProperty("__param", paramId)

            layout = QHBoxLayout()

            nameLabel = QLabel(text=param.ui_name)
            nameLabel.setToolTip(param.description)
            layout.addWidget(nameLabel)

            fieldWidget = None
            match param.param_type:
                case StrategyParamType.Number:
                    fieldWidget = QSpinBox()
                    # У Python нет способа прямо узнать максимальное/минимальное значение int.
                    # Поэтому придумываем
                    fieldWidget.setMinimum(-2**(ctypes.sizeof(ctypes.c_int) * 8 - 1) - 1)
                    fieldWidget.setMaximum(2**(ctypes.sizeof(ctypes.c_int) * 8 - 1) - 1)
                    val = self.__runConfig[paramId]
                    if not isinstance(val, int):
                        qDebug(f"Unexpected type {type(val).__name__} in field {paramId}")
                        continue
                    fieldWidget.setValue(val)
                case StrategyParamType.UNumber:
                    fieldWidget = QSpinBox()
                    fieldWidget.setMinimum(0)
                    fieldWidget.setMaximum(2**(ctypes.sizeof(ctypes.c_int) * 8 - 1) - 1)
                    val = self.__runConfig[paramId]
                    if not isinstance(val, int):
                        qDebug(f"Unexpected type {type(val).__name__} in field {paramId}")
                        continue
                    fieldWidget.setValue(val)
                case StrategyParamType.Floating:
                    fieldWidget = QDoubleSpinBox()
                    fieldWidget.setDecimals(5)
                    fieldWidget.setMinimum(sys.float_info.min)
                    fieldWidget.setMaximum(sys.float_info.max)
                    val = self.__runConfig[paramId]
                    if not isinstance(val, float):
                        qDebug(f"Unexpected type {type(val).__name__} in field {paramId}")
                        continue
                    fieldWidget.setValue(val)
                case StrategyParamType.UFloating:
                    fieldWidget = QDoubleSpinBox()
                    fieldWidget.setDecimals(5)
                    fieldWidget.setMinimum(0)
                    fieldWidget.setMaximum(sys.float_info.max)
                    val = self.__runConfig[paramId]
                    if not isinstance(val, float):
                        qDebug(f"Unexpected type {type(val).__name__} in field {paramId}")
                        continue
                    fieldWidget.setValue(val)
                case StrategyParamType.Switch:
                    fieldWidget = QComboBox()
                    for switch in param.switches:
                        fieldWidget.addItem(switch)
                    val = self.__runConfig[paramId]
                    if not isinstance(val, str):
                        qDebug(f"Unexpected type {type(val).__name__} in field {paramId}")
                        continue
                    fieldWidget.setCurrentText(val)
                case StrategyParamType.Bool:
                    fieldWidget = QCheckBox()
                    val = self.__runConfig[paramId]
                    if not isinstance(val, bool):
                        qDebug(f"Unexpected type {type(val).__name__} in field {paramId}")
                        continue
                    fieldWidget.setChecked(val)
                case _:
                    qDebug("Encountered unknown strategy param type while creating field widget")
                    continue

            fieldWidget.setObjectName("valueWidget")
            layout.addWidget(fieldWidget)
            self.__fieldWidgets[paramId] = (widget, fieldWidget)

            widget.setLayout(layout)
            self.__ui.optionsLayout.addWidget(widget)

    @Slot()
    def __saveParams(self):
        if self.__stratId is None or self.__runConfig is None:
            return

        for paramId, (_, valueWidget) in self.__fieldWidgets.items():
            match self.__runConfig.params[paramId].param_type:
                case StrategyParamType.Number:
                    if not isinstance(valueWidget, QSpinBox):
                        qDebug("Unexpected widget type")
                        continue
                    self.__runConfig[paramId] = valueWidget.value()
                case StrategyParamType.UNumber:
                    if not isinstance(valueWidget, QSpinBox):
                        qDebug("Unexpected widget type")
                        continue
                    self.__runConfig[paramId] = valueWidget.value()
                case StrategyParamType.Floating:
                    if not isinstance(valueWidget, QDoubleSpinBox):
                        qDebug("Unexpected widget type")
                        continue
                    self.__runConfig[paramId] = valueWidget.value()
                case StrategyParamType.UFloating:
                    if not isinstance(valueWidget, QDoubleSpinBox):
                        qDebug("Unexpected widget type")
                        continue
                    self.__runConfig[paramId] = valueWidget.value()
                case StrategyParamType.Switch:
                    if not isinstance(valueWidget, QComboBox):
                        qDebug("Unexpected widget type")
                        continue
                    self.__runConfig[paramId] = valueWidget.currentText()
                case StrategyParamType.Bool:
                    if not isinstance(valueWidget, QCheckBox):
                        qDebug("Unexpected widget type")
                        continue
                    self.__runConfig[paramId] = valueWidget.isChecked()
                case _:
                    qDebug("Encountered unknown strategy param type while saving field data")
                    continue

        self.optionsApplied.emit(self.__runConfig)
        self.hide()
