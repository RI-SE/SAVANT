"""Dialog for verifying manifest files."""

from pathlib import Path

from PyQt6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QFileDialog,
    QMessageBox,
    QGroupBox,
    QTextEdit,
    QProgressBar,
)

from ...services.manifest_service import ManifestService, VerificationResult


class VerifyManifestDialog(QDialog):
    """Dialog for verifying manifest files against source files."""

    def __init__(self, parent=None):
        super().__init__(parent)

        self.manifest_service = ManifestService()

        self.setWindowTitle("Verify Manifest")
        self.setMinimumWidth(600)
        self.setMinimumHeight(400)

        self._setup_ui()

    def _setup_ui(self):
        """Setup the UI."""
        layout = QVBoxLayout(self)

        # File selection
        file_group = QGroupBox("Manifest File")
        file_layout = QHBoxLayout(file_group)

        self.file_edit = QLineEdit()
        self.file_edit.setPlaceholderText("Select a manifest JSON file...")
        self.file_edit.textChanged.connect(self._on_file_changed)
        file_layout.addWidget(self.file_edit)

        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse_file)
        file_layout.addWidget(browse_btn)

        layout.addWidget(file_group)

        # Verify button
        btn_layout = QHBoxLayout()
        self.verify_btn = QPushButton("Verify")
        self.verify_btn.clicked.connect(self._on_verify)
        self.verify_btn.setEnabled(False)
        btn_layout.addStretch()
        btn_layout.addWidget(self.verify_btn)
        layout.addLayout(btn_layout)

        # Progress
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # Results
        results_group = QGroupBox("Verification Results")
        results_layout = QVBoxLayout(results_group)

        # Summary labels
        summary_layout = QFormLayout()
        self.total_label = QLabel("-")
        self.ok_label = QLabel("-")
        self.changed_label = QLabel("-")
        self.missing_label = QLabel("-")
        self.status_label = QLabel("-")

        summary_layout.addRow("Total files:", self.total_label)
        summary_layout.addRow("Files OK:", self.ok_label)
        summary_layout.addRow("Files changed:", self.changed_label)
        summary_layout.addRow("Files missing:", self.missing_label)
        summary_layout.addRow("Status:", self.status_label)

        results_layout.addLayout(summary_layout)

        # Details text
        self.details_text = QTextEdit()
        self.details_text.setReadOnly(True)
        self.details_text.setPlaceholderText("Verification details will appear here...")
        results_layout.addWidget(self.details_text)

        layout.addWidget(results_group)

        # Close button
        close_layout = QHBoxLayout()
        close_layout.addStretch()
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        close_layout.addWidget(close_btn)
        layout.addLayout(close_layout)

    def _browse_file(self):
        """Browse for manifest file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Manifest File", "", "JSON Files (*.json);;All Files (*)"
        )
        if file_path:
            self.file_edit.setText(file_path)

    def _on_file_changed(self, text):
        """Handle file path change."""
        path = Path(text.strip())
        self.verify_btn.setEnabled(path.exists() and path.suffix == ".json")

    def _on_verify(self):
        """Handle verify button click."""
        manifest_path = Path(self.file_edit.text().strip())

        if not manifest_path.exists():
            QMessageBox.warning(self, "Error", "Manifest file not found.")
            return

        # Show progress
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate
        self.verify_btn.setEnabled(False)

        # Run verification
        try:
            result = self.manifest_service.verify_manifest(manifest_path)
            self._show_results(result)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Verification failed: {e}")
        finally:
            self.progress_bar.setVisible(False)
            self.verify_btn.setEnabled(True)

    def _show_results(self, result: VerificationResult):
        """Display verification results."""
        self.total_label.setText(str(result.total_files))
        self.ok_label.setText(str(result.files_ok))
        self.changed_label.setText(str(len(result.files_changed)))
        self.missing_label.setText(str(len(result.files_missing)))

        if result.all_valid:
            self.status_label.setText("✓ All files verified successfully")
            self.status_label.setStyleSheet("color: green; font-weight: bold;")
        else:
            self.status_label.setText("✗ Issues found")
            self.status_label.setStyleSheet("color: red; font-weight: bold;")

        # Build details text
        details = []

        if result.files_changed:
            details.append("Changed files:")
            for msg in result.files_changed:
                details.append(f"  • {msg}")
            details.append("")

        if result.files_missing:
            details.append("Missing files:")
            for msg in result.files_missing:
                details.append(f"  • {msg}")
            details.append("")

        if result.all_valid:
            details.append("All source files match their recorded hashes.")

        self.details_text.setText("\n".join(details))
