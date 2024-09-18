from PySide6.QtCore import (
    Qt
)

from PySide6.QtWidgets import (
    QApplication,
    QStyleFactory
)

from Frameworks_interface.mainwindow import (
    init_config_app,
    loader_settings,
    MainWindow
)

import sys

# [1.5]
'''
    @brief  Точка входа в приложение.
'''
if __name__ == "__main__":
    init_config_app()  # Инициализация метаданных приложения для возможности обращения по ним к реестру ОС
    # и получение информации о текущей теме
    app = QApplication(sys.argv)
    app.setStyle(QStyleFactory.create("Fusion"))
    [qAppStyle, current_theme] = loader_settings()
    mw = MainWindow(qAppStyle, current_theme)
    mw.show()
    sys.exit(app.exec())
