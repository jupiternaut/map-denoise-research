# Before-outcome correction

The scan55_roi0 native smoke run finished before evaluation-reference access.
Independent audit then found that duplicate reference UV coordinates could make
kNN return another row before self, so blindly dropping neighbour zero retained
a self-edge. Potts energy assigned zero to it while a synchronous update imposed
spurious label inertia. The final implementation explicitly excludes self and
keeps six other neighbours; the duplicate-UV regression test was added.
The earlier smoke directory is retained unchanged and is not evaluation input.
Full-scene inference uses the repaired graph. No thresholds changed.

Unsupported rows stay KEEP. Their geometric neighbours can still receive a
KEEP preference through the graph; this is an explicit smoothness prior, not a
new observation. All field costs use one four-candidate common-view mask.
