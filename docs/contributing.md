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
## Local Checks

Run the following before opening a pull request:

```bash
hatch check --fix    # format, lint, and type check
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
5. Ensure all local checks pass (`hatch check --fix`, `hatch run security`, `hatch test -c`).
6. Provide a clear PR description with motivation and scope.

## Reporting Issues

Please include:

- Expected behavior and actual behavior.
- A minimal reproducible example.
- Python version, OS, and package versions.
- Relevant logs or stack traces.
