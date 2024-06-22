import sys
import csv
import argparse
from models import AffinityLM
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QPushButton, QTextEdit, QListWidget, QFileDialog, 
                             QCheckBox, QLineEdit, QFrame, QProgressBar, QStackedWidget)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont


class StylesheetManager:
    def __init__(self):
        self.color_palette = {
            "bg_main": "#f2eaff",        # Slightly muted light purple
            "bg_secondary": "#e8e0fa",   # Muted secondary background
            "bg_input": "#faf7ff",       # Very light purple for input backgrounds
            "text_primary": "#3a3a3a",   # Dark gray, not pure black
            "text_light": "#f9f9f9",     # Off-white for light text
            "accent_primary": "#8a65d7", # Slightly muted purple
            "accent_secondary": "#a590d0", # Lighter muted purple
            "accent_hover": "#7c58c9",   # Darker purple for hover states
            "accent_pressed": "#6c4ab5", # Even darker purple for pressed states
            "highlight": "#b19cd9",      # Purple highlight color
            "highlight_hover": "#cebae6" # Lighter purple for highlight hover
        }
        
        with open("style.txt", "r") as f:
            style = f.read()

        self.stylesheet_template = style

    def get_stylesheet(self):
        return self.stylesheet_template.format(**self.color_palette)

    def update_color(self, color_name, color_value):
        if color_name in self.color_palette:
            self.color_palette[color_name] = color_value
        else:
            raise ValueError(f"Color '{color_name}' not found in palette.")

class BindwellApp(QMainWindow):
    def __init__(self, device = "cpu"):
        super().__init__()
        self.setWindowTitle("BINDWELL")
        self.setGeometry(100, 100, 1000, 800)

        self.device = device

        sm = StylesheetManager()
        self.setStyleSheet(sm.get_stylesheet())

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)

        self.create_title()
        self.create_stacked_widget()
        self.create_loading_screen()
        self.create_content()


        self.csv_filename = None
        self.model = None

        # Start with loading screen
        self.stacked_widget.setCurrentIndex(0)

        QTimer.singleShot(100, self.load_model)

    def create_title(self):
        title_label = QLabel("BINDWELL")
        title_label.setObjectName("title")
        title_label.setAlignment(Qt.AlignCenter)
        self.main_layout.addWidget(title_label)

    def create_stacked_widget(self):
        self.stacked_widget = QStackedWidget()
        self.main_layout.addWidget(self.stacked_widget)

    def create_loading_screen(self):
        loading_widget = QWidget()
        loading_layout = QVBoxLayout(loading_widget)

        loading_label = QLabel("Loading model...")
        loading_label.setAlignment(Qt.AlignCenter)
        loading_label.setFont(QFont('Segoe UI', 14))
        loading_layout.addWidget(loading_label)

        self.stacked_widget.addWidget(loading_widget)

    def create_content(self):
        content_widget = QWidget()
        content_layout = QHBoxLayout(content_widget)
        content_layout.setSpacing(15)
        
        self.create_input_widgets(content_layout)
        self.create_output_widgets(content_layout)

        self.stacked_widget.addWidget(content_widget)

    def load_model(self):
        # This takes a while to load.
        self.model = AffinityLM(device=self.device)
        # Switch to main content
        self.stacked_widget.setCurrentIndex(1)

    def create_input_widgets(self, parent_layout):
        input_layout = QVBoxLayout()
        parent_layout.addLayout(input_layout)

        seq_label = QLabel("Target Protein")
        seq_label.setFont(QFont('Segoe UI', 14))
        seq_label.setAlignment(Qt.AlignCenter)
        input_layout.addWidget(seq_label)

        self.seq_entry = QTextEdit()
        self.seq_entry.setPlaceholderText("FASTA sequence...")
        self.seq_entry.setLineWrapMode(QTextEdit.WidgetWidth)
        self.seq_entry.setAcceptRichText(False)
        input_layout.addWidget(self.seq_entry)

        self.run_button = QPushButton("Run Screening")
        self.run_button.clicked.connect(self.run_screening)
        input_layout.addWidget(self.run_button)

        self.advanced_check = QCheckBox("Show Advanced Settings")
        self.advanced_check.stateChanged.connect(self.toggle_advanced)
        input_layout.addWidget(self.advanced_check)

        self.advanced_frame = QFrame()
        self.advanced_frame.setObjectName("advanced_frame")
        self.advanced_frame.setVisible(False)
        self.create_advanced_settings(self.advanced_frame)
        input_layout.addWidget(self.advanced_frame)

    def create_advanced_settings(self, parent):
        advanced_layout = QVBoxLayout(parent)

        file_label = QLabel("Upload Custom Drug Database:")
        advanced_layout.addWidget(file_label)

        self.file_button = QPushButton("Browse")
        self.file_button.clicked.connect(self.load_file)
        advanced_layout.addWidget(self.file_button)

        self.file_label = QLabel("No file selected")
        advanced_layout.addWidget(self.file_label)

        batch_sizes = [
            ("Protein encoding batch size:", "protein_batch", "2"),
            ("Molecule encoding batch size:", "molecule_batch", "16"),
            ("Prediction module batch size:", "prediction_batch", "1024")
        ]
        for text, name, default_value in batch_sizes:
            label = QLabel(text)
            advanced_layout.addWidget(label)
            entry = QLineEdit(default_value)
            advanced_layout.addWidget(entry)
            setattr(self, f"{name}_entry", entry)

        self.caching_check = QCheckBox("Save Cache")
        self.caching_check.setChecked(False)  # Caching on by default
        advanced_layout.addWidget(self.caching_check)

    def create_output_widgets(self, parent_layout):
        output_layout = QVBoxLayout()
        parent_layout.addLayout(output_layout)

        title_label = QLabel("Drug Candidates")
        title_label.setFont(QFont('Segoe UI', 14))
        title_label.setAlignment(Qt.AlignCenter)
        output_layout.addWidget(title_label)

        self.result_list = QListWidget()
        self.result_list.setSpacing(3)
        output_layout.addWidget(self.result_list)

    def toggle_advanced(self, state):
        self.advanced_frame.setVisible(state == Qt.Checked)

    def load_file(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Open CSV File", "", "CSV Files (*.csv)")
        if filename:
            self.file_label.setText(filename.split("/")[-1])
            self.csv_filename = filename

    def get_smiles_from_csv(self, filename):
        smiles_list = []
        with open(filename, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            if 'smiles_can' in reader.fieldnames:
                smiles_column = 'smiles_can'
            elif 'smiles' in reader.fieldnames:
                smiles_column = 'smiles'
            else:
                raise ValueError("CSV file does not contain 'smiles_can' or 'smiles' column")
            
            for row in reader:
                smiles_list.append(row[smiles_column])
        return smiles_list

    def run_screening(self):
        protein_sequence = self.seq_entry.toPlainText().strip()
        
        if self.csv_filename:
            csv_path = self.csv_filename
        else:
            csv_path = "data/drugs.csv"
        
        try:
            smiles_list = self.get_smiles_from_csv(csv_path)
        except Exception as e:
            self.result_list.clear()
            self.result_list.addItem(f"Error reading CSV: {str(e)}")
            return

        protein_batch = int(self.protein_batch_entry.text() or 0)
        molecule_batch = int(self.molecule_batch_entry.text() or 0)
        prediction_batch = int(self.prediction_batch_entry.text() or 0)
        save_cache = self.caching_check.isChecked()
        # You would use these values in your actual screening function

        results = self.model.score_molecules(protein=protein_sequence,molecules=smiles_list, batch_size=prediction_batch, prot_batch_size=protein_batch, mol_batch_size=molecule_batch, save_cache=save_cache).to_numpy()

        self.result_list.clear()
        for i, (drug, score) in enumerate(results[:100], 1):
            self.result_list.addItem(f"{score:<6.2f} {drug}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BINDWELL - Drug Screening Application")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], 
                        help="Specify the device to run the model on (cpu or cuda)")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    window = BindwellApp(device=args.device)
    window.show()
    sys.exit(app.exec_())
