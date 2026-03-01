"""ELU baseline is intentionally disabled: ELU is not globally convex over R."""


ELU_SKIP_REASON = "Skipped ELU-ICNN: ELU is monotone but not globally convex on R, violating ICNN convexity assumptions."
