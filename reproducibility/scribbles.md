# Scribble annotations

ACDC uses supplied expert scribbles, including background traces. BTCV_cervix and PROMISE12 use ScribbleBench-generated annotations. Fixed annotation files define the supervision used by the experiments; the preprocessing entry point converts those files rather than regenerating curves.

## Generator configuration

The available ScribbleBench configuration and implementation specify the following settings for both foreground and background. This describes the generator configuration, rather than a verified per-case generation record.

| Setting | Value |
|---|---|
| NURBS curves per selected component | 1 |
| Curve degree / control points / weights | 2 / 4 / all 1.0 |
| Curve evaluation increment | 0.001 in parameter space |
| Knot vector | Generated automatically from degree and control-point count |
| Contour segments per selected component | 1 |
| Contour erosion radius | Dilation radius plus a random integer from 1 to 4 (2–5 pixels) |
| Contour length fraction | 0.07–0.15 of the eroded contour length |
| Thickness parameter | 3; disk dilation radius 1, then clipped to the component |
| Minimum component area | 0 |
| Foreground / background slice-budget factors | 1.0 / 1.0 |
| Class sampling | Equal weights among available classes; component sampling weighted by area |

The implementation uses `gen_nurbs_scribble` in ScribbleBench's `scribblebench/scribble_generation.py` and settings from `scribblebench/scribble_conf.yml`. `curve_delta` is the curve evaluation increment, not a measure of curvature. Both NURBS and contour segments are enabled in this configuration.

## Background and unknown pixels

Generated background traces lie in connected components of reference label 0, outside the annotated target classes. Foreground and background traces are dilated and clipped to their respective components. Unmarked pixels remain unknown rather than becoming background.

Source labels are:

| Dataset | Background | Foreground | Unknown |
|---|---|---|---|
| ACDC | 0 | 1–3 | 4 |
| BTCV_cervix | 0 | 1–4 | 5 |
| PROMISE12 | 0 | 1 | 2 |

For an annotated target query, its scribbles are positive, other-class and background scribbles are negative, and unknown pixels map to ignore value 255. Partial-mask losses operate only on known pixels. Background-only slices do not create matched target mask-loss terms; a local background trace is not a verified whole-slice absence label.
