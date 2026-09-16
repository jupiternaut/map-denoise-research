# Raw-row recomputation

m=32. Each learning cell =48 target draws x8 paired sample streams, not384 independent target functions.

| Family | Method | Full-domain error | Unseen-input error | Unique targets |
|---|---|---:|---:|---:|
| affine | affine | 0.000000 | 0.000000 | 27 |
| affine | exact | 0.000000 | 0.000000 | 27 |
| affine | prefix32 | 0.238932 | 0.416196 | 27 |
| nonlinear_in_grammar | affine | 0.206055 | 0.331589 | 47 |
| nonlinear_in_grammar | exact | 0.019206 | 0.126017 | 47 |
| nonlinear_in_grammar | prefix32 | 0.200846 | 0.322626 | 47 |
| out_of_grammar | affine | 0.270182 | 0.484372 | 48 |
| out_of_grammar | exact | 0.143229 | 0.491549 | 48 |
| out_of_grammar | prefix32 | 0.308919 | 0.519169 | 48 |

## Feedback at budget2

| Family | Method | Terminal error | Goals |
|---|---|---:|---:|
| affine | adaptive | 0.166667 | 30 |
| affine | fixed_prefix | 0.400000 | 30 |
| affine | open_loop | 0.166667 | 30 |
| arbitrary | adaptive | 0.195312 | 32 |
| arbitrary | fixed_prefix | 0.308594 | 32 |
| arbitrary | open_loop | 0.226562 | 32 |
| multiplexer | adaptive | 0.000000 | 48 |
| multiplexer | fixed_prefix | 0.291667 | 48 |
| multiplexer | open_loop | 0.250000 | 48 |

## Composition tie audit trigger

{"better": 199, "equal": 41142, "worse": 131, "worse_with_model_tie": 131}

All observed adaptive disadvantages coincide with predicted-risk ties; this does not establish harm from a strictly improved model objective.
