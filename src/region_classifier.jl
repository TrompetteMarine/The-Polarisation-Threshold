module RegionClassifier

using Statistics

export default_thresholds, classify_region, aggregate_robustness, region_signature_rows

function default_thresholds()
    return Dict(
        "noise_floor" => Dict(
            "reference_delta" => -0.05,
            "min_effect_on_top" => 0.02,
            "min_points" => 3,
        ),
        "jump_detector" => Dict(
            "ci_sigma" => 3.0,
            "min_jump_size" => 0.05,
        ),
        "hysteresis" => Dict(
            "ci_sigma" => 3.0,
            "min_gap" => 0.04,
            "width_min_rel" => 0.005,
        ),
        "beta_fit" => Dict(
            "min_points" => 10,
            "min_r2" => 0.985,
            "max_se" => 0.05,
            "stability_tol" => 0.06,
            "beta_I" => (0.45, 0.55),
            "beta_TC_tight" => (0.20, 0.30),
            "beta_TC_wide" => (0.30, 0.45),
        ),
        "alpha_fit" => Dict(
            "min_points" => 5,
            "min_r2" => 0.98,
            "max_se" => 0.05,
            "band_I" => (0.40, 0.60),
            "band_TC" => (0.18, 0.32),
            "band_III" => (0.00, 0.12),
        ),
        "psd_peak" => Dict(
            "min_omega_factor" => 0.2,
            "prominence_ratio" => 10.0,
            "dc_ratio" => 5.0,
            "cv_max" => 0.15,
        ),
        "reentrant" => Dict(
            "min_effect_on_top" => 0.02,
        ),
    )
end

function _in_band(x::Float64, band::Tuple{Float64,Float64})
    return isfinite(x) && x >= band[1] && x <= band[2]
end

"""
    classify_region(diag; thresholds=default_thresholds())

Returns (region_label, abstain_reason, anchor_pass).
"""
function classify_region(diag::Dict{String,Any}; thresholds::Dict = default_thresholds())
    nf = thresholds["noise_floor"]
    jump_t = thresholds["jump_detector"]
    hyst_t = thresholds["hysteresis"]
    beta_t = thresholds["beta_fit"]
    alpha_t = thresholds["alpha_fit"]
    psd_t = thresholds["psd_peak"]

    # Required diagnostics from diag
    signal_points = get(diag, "signal_points", 0)
    jump_flag = get(diag, "jump_flag", false)
    hysteresis_flag = get(diag, "hysteresis_flag", false)
    delta_kappa_rel = get(diag, "delta_kappa_rel", NaN)
    beta_mid = get(diag, "beta_mid", NaN)
    beta_wide = get(diag, "beta_wide", NaN)
    beta_tight = get(diag, "beta_tight", NaN)
    beta_mid_se = get(diag, "beta_mid_se", NaN)
    beta_mid_r2 = get(diag, "beta_mid_r2", NaN)
    beta_mid_n = get(diag, "beta_mid_n", 0)
    alpha_hat = get(diag, "alpha_hat", NaN)
    alpha_se = get(diag, "alpha_se", NaN)
    alpha_r2 = get(diag, "alpha_r2", NaN)
    alpha_n = get(diag, "alpha_n", 0)
    psd_flag = get(diag, "psd_flag", false)
    reentrant_flag = get(diag, "reentrant_flag", false)

    # Region 0 (no ordering)
    if signal_points < nf["min_points"]
        return ("Region 0", "insufficient signal points", true)
    end

    # Region V (re-entrant)
    if reentrant_flag
        return ("Region V", "", true)
    end

    # Region IV (oscillatory)
    if psd_flag
        return ("Region IV", "", true)
    end

    # Region III (subcritical)
    if jump_flag && hysteresis_flag && isfinite(delta_kappa_rel)
        if delta_kappa_rel >= hyst_t["width_min_rel"]
            if delta_kappa_rel < 0.03
                return ("Region IIIa", "", true)
            else
                return ("Region IIIb", "", true)
            end
        else
            return ("Unclassified (discontinuous)", "hysteresis width too small", false)
        end
    end

    # Continuous regions: TC or I require alpha + beta gates
    continuous_ok = !jump_flag && !hysteresis_flag
    if continuous_ok
        # Alpha gate
        alpha_ok = isfinite(alpha_hat) && isfinite(alpha_se) && isfinite(alpha_r2) &&
                   alpha_se <= alpha_t["max_se"] && alpha_r2 >= alpha_t["min_r2"] &&
                   alpha_n >= alpha_t["min_points"]
        # Beta stability gate
        beta_stable = isfinite(beta_mid) && isfinite(beta_wide) && isfinite(beta_tight) &&
                      abs(beta_wide - beta_mid) <= beta_t["stability_tol"] &&
                      abs(beta_mid - beta_tight) <= beta_t["stability_tol"]
        beta_ok = beta_mid_se <= beta_t["max_se"] && beta_mid_r2 >= beta_t["min_r2"] &&
                  beta_mid_n >= beta_t["min_points"]

        if alpha_ok && _in_band(alpha_hat, alpha_t["band_TC"])
            if _in_band(beta_tight, beta_t["beta_TC_tight"]) &&
               _in_band(beta_wide, beta_t["beta_TC_wide"]) && beta_ok
                return ("Region TC", "", true)
            else
                return ("Unclassified (continuous)", "beta gate failed for TC", false)
            end
        end

        if alpha_ok && _in_band(alpha_hat, alpha_t["band_I"]) && beta_stable && beta_ok &&
           _in_band(beta_mid, beta_t["beta_I"])
            return ("Region I", "", true)
        end

        return ("Unclassified (continuous)", "alpha/beta gate failed", false)
    end

    # If jump detected but hysteresis not confirmed
    if jump_flag
        return ("Unclassified (discontinuous)", "jump without hysteresis", false)
    end

    return ("Inconclusive", "robustness disagreement", false)
end

"""
    aggregate_robustness(rows)

rows: Vector of Dicts with keys impl, dt, N, region_label, anchor_pass.
Returns final label, confidence, abstain_reason.
"""
function aggregate_robustness(rows::Vector{Dict{String,Any}})
    if isempty(rows)
        return (region_label="Inconclusive", confidence=0.0, abstain_reason="no rows")
    end

    # Group by N and vote across dt x impl (expected 4 combos)
    Ns = unique([r["N"] for r in rows])
    label_by_N = Dict{Int,String}()
    anchor_by_N = Dict{Int,Bool}()

    for N in Ns
        sub = [r for r in rows if r["N"] == N]
        counts = Dict{String,Int}()
        anchor_ok = true
        for r in sub
            lbl = r["region_label"]
            counts[lbl] = get(counts, lbl, 0) + 1
            anchor_ok &= get(r, "anchor_pass", false)
        end
        best_label = "Inconclusive"
        best_count = 0
        for (lbl, c) in counts
            if c > best_count
                best_label = lbl
                best_count = c
            end
        end
        # Require >= 3/4 agreement
        if best_count >= 3
            label_by_N[N] = best_label
            anchor_by_N[N] = anchor_ok
        else
            label_by_N[N] = "Inconclusive"
            anchor_by_N[N] = false
        end
    end

    # N agreement
    labels = collect(values(label_by_N))
    final_label = length(unique(labels)) == 1 ? labels[1] : "Inconclusive"

    # Confidence score
    conf = 0.0
    # A/B agreement: check per dt if impls match
    dt_vals = unique([r["dt"] for r in rows])
    ab_ok = true
    for dt in dt_vals
        sub = [r for r in rows if r["dt"] == dt]
        lbls = unique([r["region_label"] for r in sub])
        if length(lbls) > 1
            ab_ok = false
            break
        end
    end
    conf += ab_ok ? 0.25 : 0.0

    # dt agreement: check per impl if dt labels match
    impl_vals = unique([r["impl"] for r in rows])
    dt_ok = true
    for impl in impl_vals
        sub = [r for r in rows if r["impl"] == impl]
        lbls = unique([r["region_label"] for r in sub])
        if length(lbls) > 1
            dt_ok = false
            break
        end
    end
    conf += dt_ok ? 0.25 : 0.0

    # N agreement
    conf += (final_label != "Inconclusive") ? 0.25 : 0.0

    # Anchor diagnostic pass
    anchor_ok_all = all(values(anchor_by_N))
    conf += anchor_ok_all ? 0.25 : 0.0

    abstain = final_label == "Inconclusive" ? "robustness disagreement" : ""

    return (region_label=final_label, confidence=conf, abstain_reason=abstain)
end

function region_signature_rows()
    return [
        (Region="Region 0", Name="No ordering", Transition="None", Jump="No", Hysteresis="No",
         Delta_kappa_rel="n/a", Beta="n/a", Alpha="n/a", PSD="No", Large_kappa="Noise", Runs="coarse+refined"),
        (Region="Region I", Name="Supercritical", Transition="Continuous", Jump="No", Hysteresis="No",
         Delta_kappa_rel="n/a", Beta="0.5", Alpha="0.5", PSD="No", Large_kappa="Ordered", Runs="full"),
        (Region="Region TC", Name="Tricritical", Transition="Continuous", Jump="No", Hysteresis="No",
         Delta_kappa_rel="n/a", Beta="0.25", Alpha="0.25", PSD="No", Large_kappa="Ordered", Runs="full"),
        (Region="Region IIIa", Name="Weak subcritical", Transition="Discontinuous", Jump="Yes", Hysteresis="Yes",
         Delta_kappa_rel="0.005-0.03", Beta="n/a", Alpha="0-0.12", PSD="No", Large_kappa="Ordered", Runs="full"),
        (Region="Region IIIb", Name="Strong subcritical", Transition="Discontinuous", Jump="Yes", Hysteresis="Yes",
         Delta_kappa_rel=">=0.03", Beta="n/a", Alpha="0-0.12", PSD="No", Large_kappa="Ordered", Runs="full"),
        (Region="Region IV", Name="Oscillatory", Transition="Hopf-like", Jump="No", Hysteresis="No",
         Delta_kappa_rel="n/a", Beta="n/a", Alpha="n/a", PSD="Yes", Large_kappa="Oscillatory", Runs="full"),
        (Region="Region V", Name="Re-entrant", Transition="Two transitions", Jump="n/a", Hysteresis="n/a",
         Delta_kappa_rel="n/a", Beta="n/a", Alpha="n/a", PSD="n/a", Large_kappa="Disordered", Runs="full"),
    ]
end

end # module
