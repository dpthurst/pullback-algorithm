PYTHON = python3

check:
	$(PYTHON) -m unittest discover -s tests

m0:
	$(PYTHON) scripts/m0_theta.py

.PHONY: check m0
