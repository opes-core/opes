from pathlib import Path

TITLE_MAPPING = {
    "# `plot_wealth()`": "# `plot_wealth`",
    "# `backtest()`": "# `backtest`",
    "# `get_metrics()`": "# `get_metrics`",
    "# `base_optimizer`": "# Common Methods",
    "# `clean_weights()`":"# `clean_weights`",
    "# `optimize()`":"# `optimize`",
    "# `stats()`":"# `stats`"
}

BASE_DIR = Path(__file__).parent
INPUT_FILE = BASE_DIR / "docs" / "common_methods.md"
OUTPUT_FILE = INPUT_FILE

def refine_markdown():
    print(f"Reading raw documentation from '{INPUT_FILE}'...")
    try:
        with open(INPUT_FILE, "r", encoding="utf-16") as f:
            content = f.read()
    except UnicodeDecodeError:
        with open(INPUT_FILE, "r", encoding="utf-8") as f:
            content = f.read()
    except FileNotFoundError:
        print(f"\nERROR: The input file '{INPUT_FILE}' was not found.")
        print("Please generate it first before running this script.")
        return

    print("Replacing titles...")
    content = content.replace("# API Documentation", "")
    content = content.replace("## Classes", "")
    content = content.replace("##", "#")
    for ugly_title, pretty_title in TITLE_MAPPING.items():
        if ugly_title in content:
            assert content.count(ugly_title) == 1, f"Found more than one instance of {ugly_title}"
            print(f"  - Replacing '{ugly_title}' with '{pretty_title}'")
            content = content.replace(ugly_title, pretty_title)

    print(f"\nWriting final documentation to '{OUTPUT_FILE}'...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(content)

    print("Done. Your documentation is ready!")


if __name__ == "__main__":
    refine_markdown()