PYTHON ?= python3

.PHONY: smoke tables figures verify-checksums clean

smoke:
	$(PYTHON) reproducibility/scripts/smoke.py

tables:
	$(PYTHON) reproducibility/scripts/build_tables.py

figures:
	$(PYTHON) experiments/hidden_jacobian_routing/plot_objective_neutral_mobility_selector_figure.py

verify-checksums:
	bash reproducibility/scripts/verify_checksums.sh

clean:
	rm -rf generated analysis_outputs
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
