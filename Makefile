PYTHON   := python3.12
VENV     := .venv
PIP      := $(VENV)/bin/pip
PYTHON_V := $(VENV)/bin/python

# Backend selection: metal (default) | cuda | cpu
BACKEND  ?= metal
REQS     := requirements-$(BACKEND).txt

CUDA_FIND_LINKS := --find-links https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

.PHONY: all venv install test benchmark clean

all: install

$(VENV)/bin/activate:
	$(PYTHON) -m venv $(VENV)

venv: $(VENV)/bin/activate

install: venv
	$(PIP) install --upgrade pip
ifeq ($(BACKEND),cuda)
	$(PIP) install -r $(REQS) $(CUDA_FIND_LINKS)
else
	$(PIP) install -r $(REQS)
endif

test: install
	$(PYTHON_V) jax_test.py

benchmark: install
	$(PYTHON_V) benchmark_plot.py

clean:
	rm -rf $(VENV)
