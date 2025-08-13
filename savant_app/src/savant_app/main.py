import sys
from PyQt6.QtWidgets import QApplication
from savant_app.src.savant_app.frontend.main_window import MainWindow

if __name__ == "__main__":
    app = QApplication(sys.argv)
    # TODO - Get project name from current project
    window = MainWindow("- Highway_Test") 
    window.show()
    sys.exit(app.exec())
