# Contributing

Thank you for your interest in improving highFIS.

## Ways to Contribute

- Report bugs or request features through GitHub issues.
- Improve documentation, examples, and test coverage.
- Submit focused bug fixes and incremental feature improvements.

## Development Setup

This project uses Hatch environments for consistency.

```bash
pip install hatch
hatch env create
```

Alternative editable installation:

```bash
pip install -e .[dev]
```

## Local Checks

Run the following before opening a pull request:

```bash
hatch run typing
hatch run security
hatch test
```

## Documentation Workflow

Documentation is built with Zensical.

```bash
hatch run docs:serve
hatch run docs:build
```

## Pull Request Guidelines

1. Create a branch from `main`.
2. Keep changes small and coherent.
3. Add or update tests for behavior changes.
4. Update documentation for public API changes.
5. Provide a clear PR description with motivation and scope.

## Reporting Issues

Please include:

- Expected behavior and actual behavior.
- A minimal reproducible example.
- Python version, OS, and package versions.
- Relevant logs or stack traces.
