PYTHON = python3

check:
	$(PYTHON) -m unittest discover -s tests

m0:
	$(PYTHON) scripts/m0_theta.py

# The fuller database-wide iteration check (all sessions, n = 1..2).  Slow
# (~3.5 min) and not part of `check`: the suite covers n = 1 only.
dbcheck:
	$(PYTHON) scripts/db_iteration_check.py

.PHONY: check m0 dbcheck
