# ClickDV

GLM project linking Poisson click input data from rats to decision variable outputs.

## Project Overview

This project develops a Generalized Linear Model (GLM) to link click input data to decision variables in rat behavioral experiments. The goal is to understand how sensory inputs (clicks) relate to decision-making processes through neural activity patterns.

## Getting Started

### Installation

This project uses [uv](https://docs.astral.sh/uv/) for fast, modern Python package management.

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone <repository-url>
cd ClickDV

# Install dependencies and create virtual environment
uv sync

# Activate the environment (optional - uv run handles this automatically)
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

**Why uv?**
- 10-100x faster than pip
- Built-in virtual environment management
- Reproducible builds with `uv.lock`
- Modern Python packaging standards

### Project Structure

```
ClickDV/
├── data/               # Data storage
│   ├── raw/           # Original data files
│   └── processed/     # Processed data
├── src/               # Source code
│   ├── preprocessing/ # Data extraction and alignment
│   ├── models/        # GLM implementations
│   ├── analysis/      # Analysis and comparison tools
│   └── utils/         # Utility functions
├── notebooks/         # Jupyter notebooks for exploration
├── tests/            # Test suite
└── results/          # Output figures and reports
```

## Research Goals

1. Extract decision variables from spike data (starting with session A324)
2. Access and analyze click times from behavioral data
3. Develop GLM to link click inputs DV(t)
4. Investigate how GLM weights differ between sessions
5. Explore direct link between click inputs and decision making

## Usage

Start with the notebooks in the `notebooks/` directory for data exploration and model development.

## Development

The package uses modern Python packaging with `pyproject.toml` and uv for dependency management. Development dependencies are installed automatically.

Run commands in the virtual environment:
```bash
# Run tests
uv run pytest tests/

# Run tests with coverage
uv run pytest tests/ --cov=src --cov-report=html

# Run Python scripts
uv run python src/main.py

# Start Jupyter notebook
uv run jupyter notebook
```

Update dependencies:
```bash
# Add new dependency
uv add numpy

# Add development dependency
uv add --dev pytest

# Update all dependencies
uv sync
```