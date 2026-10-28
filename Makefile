PYTHON = python3

# The suite, split by what it exercises.  `check` is still the union of all of
# it, so nothing is lost by running the pieces.  Measured 2026-10-26 on this VM
# (shared, other sessions resident), per target: numerics ~1 s, engine ~21 s,
# database ~34 s, energies ~87 s, applet ~1 s; the whole suite ~142 s.
# test_database (34 s) and test_spines (13 s) dominate the engine group; the
# database sweep is the library-wide check, so it gets a target of its own.
ENGINE = tests.test_m0_theta tests.test_m1_residue tests.test_m2_core \
         tests.test_m2_classify tests.test_rabbit_versions tests.test_refined \
         tests.test_spines tests.test_library tests.test_session_api \
         tests.test_certificates
DATABASE = tests.test_database
ENERGIES = tests.test_energies tests.test_minimise tests.test_fastmin \
           tests.test_covermin
NUMERICS = tests.test_rational tests.test_moduli tests.test_teich \
           tests.test_thurston tests.test_rounding tests.test_mesh \
           tests.test_strictly_rational tests.test_julia

check:
	$(PYTHON) -m unittest discover -s tests

check-engine:
	$(PYTHON) -m unittest $(ENGINE)

check-database:
	$(PYTHON) -m unittest $(DATABASE)

check-energies:
	$(PYTHON) -m unittest $(ENERGIES)

check-numerics:
	$(PYTHON) -m unittest $(NUMERICS)

check-quick:
	$(PYTHON) -m unittest $(ENGINE) $(NUMERICS)

check-all-but-energies:
	$(PYTHON) -m unittest $(ENGINE) $(DATABASE) $(NUMERICS)

# the applet panels need the offscreen platform, which `check-applet` sets
check-slow: check-energies check-applet

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

.PHONY: check check-engine check-database check-energies check-numerics \
        check-quick check-all-but-energies check-slow \
        m0 dbcheck applet check-applet
