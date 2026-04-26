PYTHON   := python3.12
VENV     := .venv
PIP      := $(VENV)/bin/pip
PYTHON_V := $(VENV)/bin/python

.PHONY: all venv install test benchmark clean

all: install

$(VENV)/bin/activate:
	$(PYTHON) -m venv $(VENV)

venv: $(VENV)/bin/activate

install: venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

test: install
	$(PYTHON_V) jax_test.py

benchmark: install
	$(PYTHON_V) benchmark_plot.py

clean:
	rm -rf $(VENV)
