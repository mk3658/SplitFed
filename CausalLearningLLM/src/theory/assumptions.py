"""Documentation of theoretical assumptions underlying CRA."""

ASSUMPTIONS = {
    "A1_linear_decodability": (
        "Sensitive and spurious information is linearly decodable from Z_l. "
        "Linear probes q_s(S|Z_l) and q_c(C|Z_l) achieve above-chance performance."
    ),
    "A2_direction_existence": (
        "There exist directions v_s, v_c in representation space that capture "
        "sensitive and spurious information respectively."
    ),
    "A3_intervention_validity": (
        "Removing the projection of Z_l onto v_s approximates an intervention "
        "on S in the causal graph X->Z->Y_hat, S->Z."
    ),
    "A4_counterfactual_validity": (
        "Counterfactual examples x_cf preserve the clinical content Y while "
        "changing S or C."
    ),
    "A5_probe_space_proxy": (
        "Probe-space intervention results are directionally consistent with "
        "full model-space interventions, though magnitudes may differ."
    ),
}
