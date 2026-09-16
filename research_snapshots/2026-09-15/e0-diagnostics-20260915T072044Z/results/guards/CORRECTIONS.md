# Guard artifact wording correction

After the first complete run, a description-only correction changed the generated
`floating_point_note` in `guards.py` from "sub-ulp threshold crossings" to
"roundoff-scale threshold crossings". The measured shortfall is
5.551115123125783e-16 mm, about 10 floating-point spacings at 0.4; "sub-ulp" was
therefore inaccurate. The first-run `summary.json` is retained unchanged. Its
raw values are correct; read its phrase "sub-ulp" as "roundoff-scale".

This correction changes no fixture, threshold, selector, evaluation value, or
result. No experiment was rerun or selected in response to its outcome.

Before any test or experimental execution, an arithmetic mistake in the
single-pass test fixture was corrected: the third proposal displacement became
-6.5 mm so that its candidate pair gap is 4.5 mm (above half the initial 8 mm)
and its gap after the neighbor veto is 1.5 mm. That setup tests the prescribed
absence of iterative feedback. There are no earlier output files for that
pre-execution test correction.
