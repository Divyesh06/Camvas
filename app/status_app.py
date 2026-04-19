from PyQt5.QtCore import Qt, QPoint, pyqtSignal, QObject
from PyQt5.QtGui import QFontDatabase, QFont, QIcon
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout, QSystemTrayIcon, QMenu, qApp
)

import os
import sys

def is_frozen():
    return getattr(sys, 'frozen', False)

if not is_frozen():
    icon_path = "assets/Camvas.ico"
else:
    app_dir = os.path.dirname(sys.executable)
    icon_path = os.path.join(app_dir, 'Camvas.ico')

tray_icon = None
status_callback = None


class StateSignal(QObject):
    change_state = pyqtSignal(str)

class TrayApp(QSystemTrayIcon):
    def __init__(self, icon, window, parent=None):
        super().__init__(icon, parent)
        self.window = window
        self.setToolTip('Camvas is waiting for connection')

        menu = QMenu(parent)
        show_action = menu.addAction('Open Camvas')
        show_action.triggered.connect(self.show_window)

        exit_action = menu.addAction('Exit')
        exit_action.triggered.connect(qApp.quit)

        self.setContextMenu(menu)
        self.activated.connect(self.icon_clicked)

    def show_window(self):
        self.window.showNormal()
        self.window.raise_()
        self.window.activateWindow()

    def icon_clicked(self, reason):
        if reason == self.Trigger:  # Left click
            self.show_window()


class CustomWindow(QWidget):
    def __init__(self, state_signal):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.setFixedSize(600, 370)
        self.setStyleSheet("background-color: #06080a;")

        self.state = "waiting"
        self.is_dragging = False
        self.drag_position = QPoint()
        self.state_signal = state_signal
        self.state_signal.change_state.connect(self.set_state)

        self.load_fonts()
        self.init_ui()
        self.set_state("waiting")

    def load_fonts(self):
        QFontDatabase.addApplicationFont("Poppins-Regular.ttf")
        self.font_family = "Poppins"

    def init_ui(self):
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        self.title_bar = QWidget()
        self.title_bar.setFixedHeight(35)
        self.title_bar.setStyleSheet("background-color: #06080a;")
        title_layout = QHBoxLayout(self.title_bar)
        title_layout.setContentsMargins(10, 10, 0, 0)

        self.title_label = QLabel("Camvas")
        self.title_label.setFont(QFont(self.font_family, 11))
        self.title_label.setStyleSheet("color: white;")
        title_layout.addWidget(self.title_label)

        title_layout.addStretch()

        self.min_button = QPushButton("–")
        self.min_button.setFont(QFont(self.font_family, 14))
        self.min_button.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                color: white;
                border: none;
                padding: 5px;
            }
            QPushButton:hover {
                background-color: #666666;
            }
        """)
        self.min_button.clicked.connect(self.showMinimized)

        self.close_button = QPushButton("×")
        self.close_button.setFont(QFont(self.font_family, 14))
        self.close_button.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                color: white;
                border: none;
                padding: 5px;
            }
            QPushButton:hover {
                background-color: #ff3333;
            }
        """)
        self.close_button.clicked.connect(self.hide)

        title_layout.addWidget(self.min_button)
        title_layout.addWidget(self.close_button)

        self.content = QWidget()
        self.content_layout = QVBoxLayout(self.content)
        self.content_layout.setSpacing(6)
        self.content_layout.setContentsMargins(20, 25, 20, 30)

        self.heading = QLabel()
        self.heading.setAlignment(Qt.AlignCenter)
        self.heading.setFont(QFont(self.font_family, 16, QFont.Bold))

        self.subheading = QLabel()
        self.subheading.setAlignment(Qt.AlignCenter)
        self.subheading.setFont(QFont(self.font_family, 12))

        self.circle_button = QPushButton()
        self.circle_button.setFont(QFont(self.font_family, 16))
        self.circle_button.setFixedSize(120, 120)
        self.circle_button.clicked.connect(self.toggle_state)

        self.content_layout.addWidget(self.heading)
        self.content_layout.addWidget(self.subheading)
        self.content_layout.addStretch()
        self.content_layout.addWidget(self.circle_button, alignment=Qt.AlignCenter)
        self.content_layout.addStretch()

        self.main_layout.addWidget(self.title_bar)
        self.main_layout.addWidget(self.content)

    def toggle_state(self):
        if self.state == "started":
            new_state = "stopped"
        elif self.state == "stopped":
            new_state = "waiting"
        elif self.state == "waiting":
            new_state = "stopped"

        self.set_state(new_state)

    def set_state(self, state):
        if status_callback:
            status_callback(state)
        self.state = state
        if state == "waiting":
            
            self.heading.setText("Camvas is Waiting for Connection")
            if tray_icon:
                tray_icon.setToolTip('Camvas is waiting for connection')
            self.heading.setStyleSheet("color: #00ffaa;")
            self.subheading.setText("Select 'Camvas' as your camera in any app to start")
            self.subheading.setStyleSheet("color: #ddd;")
            self.circle_button.setText("Waiting")
            self.circle_button.setStyleSheet("""
                QPushButton {
                    background-color: #333333;
                    color: white;
                    font-size: 18px;
                    border-radius: 60px;
                    border: 2px solid #00ffaa;
                }
                QPushButton:hover {
                    background-color: #444444;
                }
            """)
        elif state == "started":
            self.heading.setText("Camvas is Streaming")
            if tray_icon:
                tray_icon.setToolTip('Camvas is streaming')
            self.heading.setStyleSheet("color: #00ffaa;")
            self.subheading.setText("Connected. Click 'Stop' to end streaming.")
            self.subheading.setStyleSheet("color: #eee;")
            self.circle_button.setText("Stop")
            self.circle_button.setStyleSheet("""
                QPushButton {
                    background-color: #e05555;
                    color: white;
                    font-size: 18px;
                    border-radius: 60px;
                    border: 2px solid white;
                }
                QPushButton:hover {
                    background-color: #e05555;
                }
            """)

        elif state == "stopped":
            self.heading.setText("Camvas is Stopped")
            if tray_icon:
                tray_icon.setToolTip('Camvas is stopped')
            self.heading.setStyleSheet("color: #00ffaa;")
            self.subheading.setText("Click 'Start' to start using Camvas again.")
            self.subheading.setStyleSheet("color: #ddd;")
            self.circle_button.setText("Start")
            self.circle_button.setStyleSheet("""
                QPushButton {
                    background-color: #2ecc71;
                    color: white;
                    font-size: 18px;
                    border-radius: 60px;
                    border: 2px solid #00ffaa;
                }
                QPushButton:hover {
                    background-color: #2ecc71;
                }
            """)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.is_dragging = True
            self.drag_position = event.globalPos()

    def mouseMoveEvent(self, event):
        if self.is_dragging:
            delta = event.globalPos() - self.drag_position
            self.move(self.x() + delta.x(), self.y() + delta.y())
            self.drag_position = event.globalPos()

    def mouseReleaseEvent(self, event):
        self.is_dragging = False


# Public controller for other scripts
class CamvasStatusApp:
    def __init__(self, callback=None, silent=False):
        global status_callback, tray_icon
        status_callback = callback
        self.app = QApplication(sys.argv)
        self.state_signal = StateSignal()
       
        self.window = CustomWindow(self.state_signal)
        self.window.setWindowIcon(QIcon(icon_path))
        self.window.setWindowTitle("Camvas")
        tray_icon = TrayApp(QIcon(icon_path), self.window)
        tray_icon.show()
        if not silent:
            self.window.show()
        

    def run(self):
        self.app.exec_()

    def set_state(self, state):
        self.state_signal.change_state.emit(state)

if __name__ == '__main__':
    app = CamvasStatusApp()
