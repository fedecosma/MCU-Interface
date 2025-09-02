import sys
from PyQt6.QtWidgets import QApplication
from gui import SerialApp

#main
if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = SerialApp()
    w.show()
    sys.exit(app.exec())
