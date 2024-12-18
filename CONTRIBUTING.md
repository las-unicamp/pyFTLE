# Contributing
---

### Running on development machine

We highly recommend using UV for better management of the project dependencies and environments.

Install library dependencies:

```console
uv sync
```

UV should create a virtual environment named `.venv` in the root of the project and install the dependencies there.


### Testing

We use pytest to run our unit tests. The development dependencies should be already installed after running `uv sync`.
It is also possible to enforce the installation with `uv sync --dev`. Pytest is required to test a local clone of pyFTLE.


You can run the whole test suite by using the following command in the base directory of the repository:

```console
uv run pytest
```

Or run specific tests with

```console
uv run pytest tests/[test]
```
