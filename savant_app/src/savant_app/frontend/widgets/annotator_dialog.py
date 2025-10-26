from PyQt6.QtWidgets import (
    QDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)


class AnnotatorDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.annotator_name = None
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Annotator Information")
        self.setMinimumWidth(300)

        layout = QVBoxLayout()

        # Label
        label = QLabel("Who is currently annotating?")
        layout.addWidget(label)

        # Text input
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Enter your name")
        layout.addWidget(self.name_input)

        # OK button
        ok_button = QPushButton("OK")
        ok_button.clicked.connect(self.accept_input)
        layout.addWidget(ok_button)

        # Allow Enter key to submit
        self.name_input.returnPressed.connect(self.accept_input)

        # Cancel button
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        layout.addWidget(cancel_button)

        self.setLayout(layout)

        # Set focus to input field
        self.name_input.setFocus()

    def accept_input(self):
        name = self.name_input.text().strip()
        if name:
            self.annotator_name = name
            self.accept()
        else:
            QMessageBox.warning(self, "Warning", "Please enter a name.")

    def get_annotator_name(self):
        return self.annotator_name

    # Example usage
    # if __name__ == '__main__':
    #    app = QApplication(sys.argv)

    #    dialog = AnnotatorDialog()
    #    result = dialog.exec_()

    #    if result == QDialog.Accepted:
    #        print(f"Annotator: {dialog.get_annotator_name()}")
    #    else:
    #        print("Dialog cancelled")

    # sys.exit()
