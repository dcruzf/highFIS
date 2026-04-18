# Contributing

Thank you for your interest in improving highFIS.

## Ways to Contribute

- Report bugs or request features through GitHub issues.
- Improve documentation, examples, and test coverage.
- Submit focused bug fixes and incremental feature improvements.

## Development Setup

This project uses [Hatch](https://hatch.pypa.io/) for environment
management, testing, formatting, and builds.

```bash
pip install hatch
```

To set up pre-commit hooks:

```bash
hatch run install
```

### Virtual Environment for VS Code

The default Hatch environment is configured to use `.venv` at the project
root. Running `hatch env create` creates it with all dev dependencies
(ruff, ty, bandit, pre-commit, etc.):

```bash
hatch env create
```

VS Code detects `.venv` automatically. If needed, select the interpreter
manually via **Ctrl+Shift+P** → *Python: Select Interpreter* →
`./.venv/bin/python`.

## Local Checks

Run the following before opening a pull request:

```bash
hatch fmt            # format and lint (ruff)
hatch run typing     # type check (ty)
hatch run security   # security scan (bandit)
hatch test -c        # tests with coverage (pytest + coverage)
```

To run all pre-commit hooks at once:

```bash
hatch run all
```

### Test Matrix

Tests run against Python 3.11, 3.12, 3.13, and 3.14. To run against a
specific version:

```bash
hatch test -py 3.12
```

### Coverage

Coverage must stay at or above **90 %**. To generate an HTML report:

```bash
hatch run html-report
```

## Documentation Workflow

Documentation is built with [Zensical](https://github.com/dcruzf/zensical).

```bash
hatch run docs:serve   # live preview
hatch run docs:build   # build static site
```

## Pull Request Guidelines

1. Create a branch from `main`.
2. Keep changes small and coherent.
3. Add or update tests for behavior changes.
4. Update documentation for public API changes.
5. Ensure all local checks pass (`hatch run all`).
6. Provide a clear PR description with motivation and scope.

## Reporting Issues

Please include:

- Expected behavior and actual behavior.
- A minimal reproducible example.
- Python version, OS, and package versions.
- Relevant logs or stack traces.
