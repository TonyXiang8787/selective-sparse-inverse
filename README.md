# Selective Sparse Inverse

A Python package for selective sparse matrix inversion.

## Development Setup

This project uses:
- **Hatchling**: Build backend
- **UV**: Python package manager and task runner
- **Ruff**: Fast linter and code formatter
- **Pytest**: Testing framework

### Prerequisites

- Python 3.9 or higher
- [UV](https://github.com/astral-sh/uv) package manager

### Installation

Install the package in development mode with dev dependencies:

```bash
uv pip install -e ".[dev]"
```

Or using UV's dependency groups:

```bash
uv pip install -e . --with dev
```

### Running Commands

#### Linting
```bash
uv run ruff check src tests
```

#### Code Formatting
```bash
uv run ruff format src tests
```

#### Running Tests
```bash
uv run pytest
```

#### Running Tests with Coverage
```bash
uv run pytest --cov=selective_sparse_inverse
```

### Project Structure

```
selective-sparse-inverse/
├── src/
│   └── selective_sparse_inverse/     # Main package
│       └── __init__.py
├── tests/                             # Test files
│   ├── conftest.py                   # Pytest configuration
│   └── test_package.py
├── pyproject.toml                     # Project configuration
└── README.md
```

### Configuration Files

- **pyproject.toml**: Package metadata, dependencies, and tool configurations
- **.python-version**: Python version specification (for pyenv)

### VS Code Setup

The `.vscode/settings.json` file configures VS Code to automatically format code with Ruff on save.

Install the [Ruff extension](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff) for better integration.

## License

MIT
