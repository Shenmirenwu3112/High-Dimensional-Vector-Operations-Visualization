import sys
from PyQt5.QtWidgets import QApplication
from frontend.ui_main import MyApp

def main():
    """
    Initializes and runs the PyQt application.
    """
    app = QApplication(sys.argv)
    main_window = MyApp()
    main_window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()