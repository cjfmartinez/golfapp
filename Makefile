SHELL := /bin/bash

VENV := venv
PYTHON := $(VENV)/bin/python
STREAMLIT := $(VENV)/bin/streamlit
APP := golfapp.py

.PHONY: run setup clean

run:
	@echo "▶ Running Streamlit app..."
	@test -x "$(STREAMLIT)" || (echo "ERROR: $(STREAMLIT) not found. Run: make setup" && exit 1)
	@$(STREAMLIT) run $(APP)

setup:
	@echo "▶ Setting up virtual environment..."
	python3 -m venv $(VENV)
	@$(PYTHON) -m pip install --upgrade pip
	@$(PYTHON) -m pip install -r requirements.txt

clean:
	@echo "▶ Removing virtual environment..."
	rm -rf venv
