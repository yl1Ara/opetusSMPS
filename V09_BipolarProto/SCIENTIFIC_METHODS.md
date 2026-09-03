# Scientific analysis methods

The online inversion viewer includes measured-range particle moments,
sulfuric-acid condensation sink, neutral Brownian coagulation sink,
lognormal modal fitting, and an experimental maximum-cross-correlation (MCC)
growth-rate cross-check.

## Scope and assumptions

- Input distributions are `dN/dlog10Dp` in `cm-3`; diameter centers are in nm.
- Number, surface, volume, CS, and CoagS use only measured finite bins. They do
  not extrapolate unmeasured particle sizes.
- Surface and volume assume spherical particles.
- CS assumes sulfuric acid condensing in air with unit accommodation.
- CoagS uses a neutral Fuchs transition-regime Brownian kernel. It excludes
  electrostatic enhancement and uses the configured spherical-particle density.
- Temperature and pressure are the configured inversion conditions. They are
  not claimed to be ambient observations unless the operator supplies measured
  ambient values.
- Positive/negative inversion labels are DMA voltage signs; selected particle
  charge has the opposite sign.
- A modal component's `area_cm3` is the fitted full-lognormal integral. The UI
  reports `fit_range_area_cm3` integrated only over measured support to avoid
  presenting extrapolated modal tails or internal gaps as measured concentration.
- MCC is an experimental distinct-estimator cross-check on the same inversion data. It is disabled by default
  and does not accept or reject the primary geometric growth tracks.

## References

- Kulmala et al. (2012), Measurement of the nucleation of atmospheric aerosol
  particles, Nature Protocols 7, 1651-1667.
  https://doi.org/10.1038/nprot.2012.091
- Fuller et al. (1966), A new method for prediction of binary gas-phase
  diffusion coefficients. https://doi.org/10.1021/ie50677a007
- Sutugin and Fuchs (1971), transition-regime correction.
  https://doi.org/10.1016/0021-8502(71)90061-9
- Hinds (1999), Aerosol Technology, 2nd edition, Brownian diffusion and
  transition-regime coagulation.
- Lampilahti et al. (2025), maximum-cross-correlation growth-rate method.
  https://doi.org/10.5194/ar-3-637-2025

The interaction and analysis design was informed by the MIT-licensed
`aerosol-functions` and `aerosol-studio` projects. See
`THIRD_PARTY_NOTICES.md`.
