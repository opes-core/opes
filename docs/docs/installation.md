# Installation

---

This page guides you through installing OPES for experimentation & research.

!!! warning "Warning:"
	OPES is currently under development. While it is relatively stable for experimentation, some features may change or break. Use at your own discretion and always verify results when testing.

## Prerequisites

- Python 3.10+ (tested up to 3.12)
- `pip` package manager

---

## Procedure

### 1. Install OPES

You can install OPES easily via PyPI:

```bash
pip install opes
```

This will fetch the latest stable release and all required dependencies.

You are also welcome to install the module directly from GitHub:

```bash
git clone https://github.com/opes-core/opes.git
cd opes-main
pip install -e .
```

!!! note "Note:"
	The `-e` flag installs OPES in editable mode, so any changes you make to the source code are reflected immediately without reinstalling. This is great for developers or those tinkering with advanced features.

---

### 2. Verify the Installation

After installation, make sure everything works by opening Python and importing OPES:

```python
>>> import opes
>>> opes.__version__
'1.0.0'
```

If no errors appear, OPES is ready to use.

---

## Dependencies

OPES requires the following Python modules:

| Module name     | Minimum version | Maximum version |
| --------------- | --------------- | --------------- |
| **NumPy**       | 2.2.6           | < 3.0           |
| **pandas**      | 2.3.3           | < 3.0           |
| **SciPy**       | 1.15.2          | < 2.0           |
| **matplotlib**  | 3.10.1          | < 4.0           |