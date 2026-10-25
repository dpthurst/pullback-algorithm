PYTHON = python3

check:
	$(PYTHON) -m unittest discover -s tests

m0:
	$(PYTHON) scripts/m0_theta.py

# The desktop applet (plan items 260, 280): the Qt window over the engine.
# It needs a display; this machine has none, so `check-applet` drives the
# same window -- construct, open a session, run every panel -- offscreen.
applet:
	PYTHONPATH=src $(PYTHON) -m pullback.app

check-applet:
	QT_QPA_PLATFORM=offscreen PYTHONPATH=src $(PYTHON) -m unittest discover -s tests -p "test_app_headless.py" -v

# The fuller database-wide iteration check (all sessions, n = 1..2).  Slow
# (~3 min) and not part of `check`: the suite covers n = 1 only.
dbcheck:
	$(PYTHON) scripts/db_iteration_check.py

.PHONY: check m0 dbcheck applet check-applet
