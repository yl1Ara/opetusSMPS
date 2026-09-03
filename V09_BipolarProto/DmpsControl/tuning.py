import math
import statistics


def lambda_pi_from_step(samples, output_low, output_high, min_response=0.5,
                        flow_min=0.0, flow_max=100.0, stability_fraction=0.08):
    """Validate an open-loop step response and return conservative Lambda PI gains."""
    if not math.isfinite(output_low) or not math.isfinite(output_high):
        raise ValueError("Step outputs must be finite")
    delta_output = output_high - output_low
    if delta_output < 0.2:
        raise ValueError("High output must be at least 0.2 V above low output")

    low = [row for row in samples if row["phase"] == "low"]
    high = [row for row in samples if row["phase"] == "high"]
    if len(low) < 5 or len(high) < 5:
        raise ValueError("Not enough fresh samples for tuning")

    for row in low + high:
        flow = float(row["flow_lpm"])
        if not math.isfinite(flow):
            raise ValueError("Nonfinite sheath-flow sample")
        if flow < flow_min or flow > flow_max:
            raise ValueError(
                f"Sheath-flow sample {flow:.3f} L/min is outside "
                f"{flow_min:.3f}..{flow_max:.3f} L/min"
            )

    tail_count = max(3, min(len(low), len(high)) // 4)
    baseline_values = [float(row["flow_lpm"]) for row in low[-tail_count:]]
    final_values = [float(row["flow_lpm"]) for row in high[-tail_count:]]
    baseline = statistics.fmean(baseline_values)
    final = statistics.fmean(final_values)

    for name, values, mean in (
        ("low", baseline_values, baseline), ("high", final_values, final)
    ):
        allowed_span = max(0.2, stability_fraction * max(abs(mean), 1.0))
        if max(values) - min(values) > allowed_span:
            raise ValueError(f"Sheath flow was not stable during the {name} plateau")

    delta_flow = final - baseline
    if abs(delta_flow) < min_response:
        raise ValueError(
            f"No usable response: flow changed only {delta_flow:.3f} L/min"
        )
    process_gain = delta_flow / delta_output
    if process_gain <= 0:
        raise ValueError("Reverse process gain: sheath flow fell when DAC output increased")

    baseline_noise = statistics.pstdev(baseline_values)
    response_threshold = baseline + max(0.05 * delta_flow, 3.0 * baseline_noise, 0.05)
    target = baseline + 0.632 * delta_flow
    response_rows = [row for row in high if float(row["flow_lpm"]) >= response_threshold]
    target_rows = [row for row in high if float(row["flow_lpm"]) >= target]
    if not response_rows:
        raise ValueError("No detectable sheath-flow response after the DAC step")
    if not target_rows:
        raise ValueError("Step response did not reach 63.2% of its final change")

    theta = max(0.0, float(response_rows[0]["step_elapsed_sec"]))
    tau = float(target_rows[0]["step_elapsed_sec"]) - theta
    if not math.isfinite(tau) or tau <= 0:
        raise ValueError("Could not determine a positive process time constant")

    lambda_seconds = max(3.0 * tau, 5.0 * theta, 1.0)
    kp = tau / (process_gain * (lambda_seconds + theta))
    integral_time = tau + theta
    ki = kp / integral_time
    if not all(math.isfinite(value) and value > 0 for value in (kp, ki)):
        raise ValueError("Calculated PI gains are invalid")

    return {
        "Kp": kp,
        "Ki": ki,
        "Kd": 0.0,
        "process_gain_lpm_per_v": process_gain,
        "tau_sec": tau,
        "theta_sec": theta,
        "lambda_sec": lambda_seconds,
        "baseline_lpm": baseline,
        "final_lpm": final,
        "delta_flow_lpm": delta_flow,
    }


def aerosol_factor_from_pressures(pressures, stability_fraction=0.10):
    """Calculate the 1 L/min calibration factor from stable positive pressure."""
    values = [float(value) for value in pressures]
    if len(values) < 5:
        raise ValueError("At least five aerosol pressure readings are required")
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Aerosol pressure readings contain nonfinite values")
    mean_pressure = statistics.fmean(values)
    if mean_pressure <= 0.01:
        raise ValueError("A stable positive aerosol pressure is required")
    allowed_span = max(0.02, stability_fraction * mean_pressure)
    if max(values) - min(values) > allowed_span:
        raise ValueError(
            "Aerosol pressure is not stable; hold external actual flow at 1.0 L/min"
        )
    return {
        "factor_lpm_per_pa": 1.0 / mean_pressure,
        "mean_pressure_pa": mean_pressure,
        "pressure_std_pa": statistics.pstdev(values),
        "sample_count": len(values),
    }
