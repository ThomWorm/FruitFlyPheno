# FruitFlyPheno

## Setup Instructions

Follow these steps to set up the environment and get started with the FruitFlyPheno project:
1. **Clone the Repository**
  ```bash
  git clone <repository-url>
  cd FruitFlyPheno
  ```

2. **Create and Activate a Virtual Environment with uv**
  - On Linux/MacOS:
    ```bash
    uv venv
    source .venv/bin/activate
    ```
  - On Windows:
    ```bash
    uv venv
    .venv\Scripts\activate
    ```

3. **Install Dependencies**
   - Install `uv`:
     ```bash
     pip install uv
     ```
   - Install project dependencies:
     ```bash
     pip install -r requirements.txt
     ```

4. **Install the Project in Editable Mode - this simplifies pathing**
   ```bash
   pip install . -e
   ```

5. **Verify Installation**
   Run the following command to ensure everything is set up correctly:
   ```bash
   fflies-model --help
   ```

You are now ready to use the FruitFlyPheno pipeline!

## Main Function

The main entry point for the FruitFlyPheno pipeline is the `main` function, located in `fflies/main.py`.

### Function Signature

```python
def main(input_json=None, plot=False, save_plot=None, print_json=False, use_pickle=False):
    """
    Main entry point for FruitFlyPheno pipeline.
    """
```

### Parameters

- **input_json** (`str` or `None`):  
  Path to input JSON file, or `"test"` to use test input. If `None`, uses test input.

- **plot** (`bool`):  
  If `True`, displays an interactive plot inline (Jupyter/Colab) or in the browser (local).

- **save_plot** (`str` or `None`):  
  If provided, saves the plot as an HTML file to this path. If `None`, does not save.

- **print_json** (`bool`):  
  If `True`, prints the output JSON to the terminal in a formatted, readable table.

- **use_pickle** (`bool`):  
  If `True`, loads/saves model results from/to a pickle file for faster plotting development.

### Usage

You can run the pipeline from the command line:

```bash
python -m fflies.main --input path/to/input.json --plot --save-plot output.html --print-json --use-pickle
```
- Use `--input_json` to specify the input JSON file 
- Use `--plot` to visualize results.
- Use `--save-plot` to save the plot as HTML.
- Use `--print-json` to print the output JSON in a readable table.
- Use `--use-pickle` to cache results for faster development.

For more details, see the docstring in `fflies/main.py`.



## Sample commands
uv run dashboard.py ../outputs/medfly_alameda_results.nc

run main.py --input '../config/alameda_test.json' --output-path '../outputs/' --exec-dashboard
