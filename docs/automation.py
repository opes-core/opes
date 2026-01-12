import subprocess
from pathlib import Path

# Obtaining the script directory
BASE_DIR = Path(__file__).parent

# Define paths relative to script location, input & output
input_dir = BASE_DIR.parent / "opes"
output_dir = BASE_DIR / "docs"

# Hardcoding title mappings. Fine for small documentation
TITLE_MAPPING = {
    "# `plot_wealth()`": "# `plot_wealth`",
    "# `backtest()`": "# `backtest`",
    "# `get_metrics()`": "# `get_metrics`",
    "# `base_optimizer`": "# Common Methods",
    "# `regularizer`": "# Regularization",
    "# `clean_weights()`": "# `clean_weights`",
    "# `optimize()`": "# `optimize`",
    "# `stats()`": "# `stats`",
    "# `set_regularizer()`": "# `set_regularizer`",
}

# Hardcoding paths. Selected files are only modified to avoid unnecessary confusion
PATHS = {
    input_dir / "backtester.py": output_dir / "backtesting.md",
    input_dir / "regularizer.py": output_dir / "regularization.md",
    input_dir / "objectives/distributionally_robust.py": output_dir / "objectives/dro.md",
    input_dir / "objectives/heuristics.py": output_dir / "objectives/heuristics.md",
    input_dir / "objectives/markowitz.py": output_dir / "objectives/markowitz.md",
    input_dir / "objectives/online.py": output_dir / "objectives/online_learning.md",
    input_dir / "objectives/risk_measures.py": output_dir / "objectives/risk_measures.md",
    input_dir / "objectives/utility_theory.py": output_dir / "objectives/utility_theory.md",
}

# Function to get markdown documentation
# uses pymarkdoc command to extract docstrings from input file into the output file
def get_markdown_documentation():

    # Execute PowerShell command for each pair
    for key, value in PATHS.items():
        try:
            # Convert Path objects to strings
            key_str = str(key)
            value_str = str(value)

            print(f"EXECUTING COMMAND ON {value_str}")
            # Running the command
            result = subprocess.run(
                ["powershell", "-Command", f"pymarkdoc '{key_str}' > '{value_str}'"],
                capture_output=True,
                text=True,
                check=True,
            )
            print(f"SUCCESS: {key.name} -> {value.name}")
        except subprocess.CalledProcessError as e:
            print(f"ERROR: {key.name} - {e.stderr}")

# Documentation post processing 
# Swaps titles and changes minor formatting for beautification and readablity
# Changes are hardcoded
def refine_markdown_documentation():

    # Iterating through values
    # Since the files are modified in place, input and output files are the same
    for value in PATHS.values():

        # Input and output files
        INPUT_FILE = value
        OUTPUT_FILE = INPUT_FILE

        # Logging action
        print(f"READING DOCUMENTATION FROM'{INPUT_FILE}'...")

        # Trying to open in utf-16, pymarkdoc writes in utf-16
        # If it fails, then it tries on utf-8
        try:
            with open(INPUT_FILE, "r", encoding="utf-16") as f:
                content = f.read()
        except UnicodeDecodeError:
            with open(INPUT_FILE, "r", encoding="utf-8") as f:
                content = f.read()
        except FileNotFoundError:
            raise FileNotFoundError(f"\nERROR: The input file '{INPUT_FILE}' was not found.")

        # Changing necessary content
        print("CHANGING CONTENT")

        # ----- CHANGES -----

        # Formatting changes
        content = content.replace("# API Documentation", "")
        content = content.replace("## Classes", "")
        content = content.replace("## ", "# ")

        # Title Changes
        for ugly_title, pretty_title in TITLE_MAPPING.items():
            if ugly_title in content:
                print(f"  - Replacing '{ugly_title}' with '{pretty_title}'")
                content = content.replace(ugly_title, pretty_title)

        # -------------------

        # Writing markdown to output file in utf-8 format
        # Mkdocs uses utf-8 markdown
        print(f"WRITING DOCUMENTATION TO '{OUTPUT_FILE}'...")
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            f.write(content)

    # Finishing Message
    print("PROCESSING FINISHED")


if __name__ == "__main__":
    get_markdown_documentation()
    refine_markdown_documentation()