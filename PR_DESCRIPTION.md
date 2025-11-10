This PR introduces a CI/CD pipeline to automate the building and publishing of the `pyftle` package to PyPI and Docker Hub.

### Summary of Changes

-   **GitHub Actions Workflow:** A new workflow has been added at `.github/workflows/publish.yaml`. This workflow is triggered on every push to the `main` branch and consists of two jobs:
    -   `pypi-publish`: Builds the Python package and publishes it to PyPI.
    -   `docker-publish`: Builds a Docker image and publishes it to Docker Hub.
-   **Dockerfile:** A `Dockerfile` has been added to the root of the project. This file defines the environment for building and running the `pyftle` application in a Docker container.
-   **`pyproject.toml`:** The `pyproject.toml` file has been updated to include a new script for the main application (`pyftle`) and to make the `scripts` directory a package.
-   **Test Refactoring:** Common fixtures have been moved to `tests/conftest.py`, and test files have been updated to use these centralized fixtures.
-   **Code Comment Removal:** All code comments have been removed from the test files for conciseness.
-   **Configuration Migration:**
    -   `pytest.ini` configuration has been migrated to `[tool.pytest.ini_options]` in `pyproject.toml`. The `pytest.ini` file has been deleted.
    -   `.ruff.toml` configuration has been migrated to `[tool.ruff]` in `pyproject.toml`. The `.ruff.toml` file has been deleted.
-   **README Update:** Instructions for running the `pyftle` application using the public Docker image from Docker Hub have been added to `README.md`.

### Local Testing

The following local tests were performed to ensure the Docker image and the refactored tests are working correctly:

-   **Docker Build:** The Docker image was successfully built using the provided `Dockerfile` after all modifications. This process included the installation of all necessary dependencies, including the C++ extension.
-   **`pyftle --help`:** The `pyftle --help` command was executed inside the Docker container and it returned the correct help message, confirming that the main script is executable and the entrypoint is correctly configured.
-   **`generate-stubs --help`:** The `generate-stubs` command was executed inside the Docker container and it ran successfully, confirming that the `dev` dependencies are correctly installed and the script is executable.
-   **Test Execution:** All 53 local tests passed successfully after refactoring the fixtures and removing comments.
-   **Docker Entrypoint Re-verification:** After migrating `pytest.ini` and `.ruff.toml` to `pyproject.toml`, the Docker image was rebuilt and the `pyftle --help` command was re-executed successfully, confirming the entrypoint still works.

### Configuration

To enable the new GitHub Actions workflow, you need to configure the following:

1.  **PyPI Trusted Publishing:** To allow the workflow to publish to PyPI, you need to configure trusted publishing for your project on PyPI. You can find detailed instructions on how to do this in the [PyPI documentation](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/).

2.  **Docker Hub Secrets:** To allow the workflow to publish to Docker Hub, you need to add the following secrets to your GitHub repository settings:
    -   `DOCKERHUB_USERNAME`: Your Docker Hub username.
    -   `DOCKERHUB_TOKEN`: A Docker Hub access token. You can create one in your Docker Hub account settings.

Once these configurations are in place, the workflow will automatically build and publish your package and Docker image whenever you merge changes into the `main` branch.
