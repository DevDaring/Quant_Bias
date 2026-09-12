"""mixed_study -- the integrated compression-to-outcome study.

Joins two sibling projects into one testable chain

    compression operation -> local residual -> propagation -> answer-score
    change -> task damage / stereotype-aligned error / group disparity

`quantbias` (../quant-bias) supplies adapters, quantizers, tasks, metrics and
the E0-E7 records; `living-inference` supplies the local error-accounting
framework and its perturbation profiles, read only. Nothing here rewrites a
source result file: corrected analyses go under results/v2/.

Implements Next_Plan.md sections 3 (measurement corrections), 5 (B1 matched
residual-source and direction experiment), 6 (task-relevant prediction) and
the P0/P1 work packages of section 10.
"""

from . import common as _common  # noqa: F401,E402  -- registers ../quant-bias on sys.path
