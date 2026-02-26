# CFD-PINN-Study: Setup & Usage Guide

This repository contains a suite of Physics-Informed Neural Network (PINN) examples for computational fluid dynamics (CFD) problems. The main entry point is the `Combined_PINN_Suite.ipynb` Jupyter notebook.

## Prerequisites
- **Python 3.8+** (recommended)
- **VS Code** with the [Jupyter extension](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter)

## Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone <repo-url>
   cd CFD-PINN-Study
   ```

2. **Open in VS Code**
   - Launch VS Code.
   - Open the `CFD-PINN-Study` folder.

3. **Create a Python Environment**
   - Open the VS Code command palette (`Ctrl+Shift+P`), type `Python: Create Environment`, and follow the prompts.
   - Alternatively, create a virtual environment manually:
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```

4. **Install Required Packages**
   - Open a terminal in VS Code and run:
     ```bash
     pip install -r requirements.txt
     ```
   - If `requirements.txt` is missing, install common packages:
     ```bash
     pip install numpy scipy matplotlib torch jupyter
     ```

5. **Install Jupyter Extension (if not already installed)**
   - Go to Extensions (`Ctrl+Shift+X`) and search for "Jupyter". Install the extension by Microsoft.

## Running the Notebook

1. In VS Code, open `Combined_PINN_Suite.ipynb`.
2. Select the Python interpreter matching your environment (look for `.venv` or your chosen environment).
3. Run cells sequentially or use "Run All" to execute the notebook.

## Notes
- Example problem codes are in the `problem_codes/` directory.
- For best results, ensure all dependencies are installed in your environment.
- If you encounter missing package errors, install them using `pip install <package-name>`.

---

For questions or issues, please open an issue in this repository.
