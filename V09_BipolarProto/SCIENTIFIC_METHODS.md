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
- Temperature and pressure can be matched per scan from an optional ambient CSV
  containing `time`, `temperature_k` or `temperature_c`, and `pressure_pa` or
  `pressure_hpa`. Observations outside the configured nearest-match tolerance
  fall back explicitly to the configured inversion conditions.
- Positive/negative inversion labels are DMA voltage signs; selected particle
  charge has the opposite sign.
- The ion-mobility ratio is always `Zn/Zp`: negative-ion mobility divided by
  positive-ion mobility. The nominal default is 1.60/1.35.
- A scan-derived `Zn/Zp` uses the Gunn-Woessner square-root relation and is
  corrected by the configured `Np/Nn`. It is not used for the experimental
  Fuchs-type inversion, which uses the configured ratio. An optional additive
  operator offset and temporal step limiter are shown separately from the raw
  estimate; the additive offset defaults to zero for new settings.
- The optional polarity-consistency diagnostic pairs positive- and
  negative-voltage columns by scan ID and compares only common measured support,
  defaulting to 20-70 nm. Under the singly charged Gunn-Woessner relation it
  reports `effective Zn/Zp = used Zn/Zp * sqrt(median(Npositive/Nnegative))`
  and evaluates the corresponding robust log-mismatch over a candidate-ratio
  grid. This is a model-dependent consistency estimate, not an independent ion
  mobility measurement. It can absorb `Np/Nn`, aerosol evolution between
  polarity scans, CPC response and efficiency, DMA asymmetry, transport losses,
  multiply charged particles, and other inversion errors. Results are therefore
  never applied automatically. Per-CPC summaries and correction-mode provenance
  are exported so instrument-dependent estimates can be identified. The
  candidate objective first takes a median over diameter within each pair and
  then a median over equally weighted pairs. An optimum at the configured
  candidate-range boundary is explicitly marked as bound-limited.
- The experimental Fuchs-type solver is a corrected Chen-derived limiting-
  sphere implementation with adaptive charge-state detailed balance. It is not
  yet claimed to reproduce the complete Hoppel-Frick (1986) three-body model;
  like-sign attachment can truncate to zero below about 20 nm.
- A modal component's `area_cm3` is the fitted full-lognormal integral. The UI
  reports `fit_range_area_cm3` integrated only over measured support to avoid
  presenting extrapolated modal tails or internal gaps as measured concentration.
- MCC is an experimental distinct-estimator cross-check on the same inversion data. It is disabled by default
  and does not accept or reject the primary geometric growth tracks.
- Particle formation diagnostics report a three-term apparent budget in
  `cm-3 s-1`:
  accumulation `dN/dt`, growth outflux `GR*N/(d2-d1)`, and neutral Brownian
  restricted neutral Brownian coagulation sink approximation. It omits
  smaller-collector interactions and coagulation-product gains. Apparent `J` is reported only
  where all three terms are finite; it does not include dilution, deposition,
  transport, or charged-particle terms. Growth-slope p10-p90 propagation is a
  sensitivity range, not a confidence interval.
- The experimental Fuchs-type path treats configured ion mobilities as values
  at the scan conditions. Pressure is validated and recorded but does not
  independently rescale the charging fractions.

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
- Fuchs (1963), stationary aerosol charge distribution in a bipolar ionic
  atmosphere. https://doi.org/10.1007/BF01993343
- Hoppel and Frick (1986), ion-aerosol attachment coefficients and steady-state
  bipolar charge distributions. https://doi.org/10.1080/02786828608959073
- Wiedensohler (1988), polynomial approximation of bipolar submicron charge
  distributions. https://doi.org/10.1016/0021-8502(88)90278-9

The interaction and analysis design was informed by the MIT-licensed
`aerosol-functions` and `aerosol-studio` projects. See
`THIRD_PARTY_NOTICES.md`.
