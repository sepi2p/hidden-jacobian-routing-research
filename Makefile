PYTHON ?= python3

.PHONY: smoke tables figures gtsrb-aggregate verify-checksums clean

smoke:
	$(PYTHON) reproducibility/scripts/smoke.py

tables:
	$(PYTHON) reproducibility/scripts/build_tables.py

figures:
	$(PYTHON) experiments/hidden_jacobian_routing/plot_objective_neutral_mobility_selector_figure.py

gtsrb-aggregate:
	$(PYTHON) experiments/eaai_gtsrb/aggregate_gtsrb_replications.py \
		--base-dir analysis_outputs/eaai_gtsrb \
		--output-dir analysis_outputs/eaai_gtsrb/aggregate \
		--figure-dir generated/gtsrb/figures \
		--table-dir generated/gtsrb/tables

verify-checksums:
	bash reproducibility/scripts/verify_checksums.sh

clean:
	rm -rf generated analysis_outputs
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
