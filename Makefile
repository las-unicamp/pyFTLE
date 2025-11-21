.PHONY: docs generate-stubs

# Build Sphinx documentation using the Makefile inside docs/
docs:
	@echo "Building documentation using docs/Makefile..."
	@uv run make -C docs docs

# Generate Python type stubs
generate-stubs:
	uv run python scripts/generate_stubs.py
