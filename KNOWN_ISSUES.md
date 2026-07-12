# Known Issues

## Avoidance artifacts in releases through v1.2.2

The frozen `mechanism_breaking` summaries in releases through `v1.2.2` were produced by sorting the 400-image final-test split by class and applying `head(200)`. Since the exact split contains 40 images per CIFAR-10 class, those summaries contain classes 0--4 only.

Consequences:

- the published 200-image avoidance values are not ten-class estimates;
- they must not be used to support a general CIFAR-10 functional-reliance claim;
- the released 20-step, three-restart sign-PGD optimizer and dimension-only Gaussian control are also insufficient for the strongest causal interpretation.

The producer now enforces class-balanced selection. A replacement experiment is in progress with all ten classes, adaptive constrained optimization, `L_inf`-geometry-matched controls, and functional coordinate-reparameterization tests. The affected frozen summaries are retained only for audit provenance and will be superseded, not silently rewritten.
