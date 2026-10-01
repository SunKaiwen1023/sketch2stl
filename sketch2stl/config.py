"""Every magic number lives here. If you are about to hard-code a tolerance
anywhere else in the codebase, put it here instead and import it.
"""

# --- canvas <-> model space ------------------------------------------------ #
# The single conversion point between screen pixels and millimetres.
# The drawing canvas is CANVAS_W x CANVAS_H pixels and maps to a
# (CANVAS_W / PX_PER_MM) x (CANVAS_H / PX_PER_MM) millimetre sheet.
CANVAS_W = 640
CANVAS_H = 480
PX_PER_MM = 4.0                 # 640px / 4 = 160 mm wide sheet

# --- stroke preprocessing -------------------------------------------------- #
RESAMPLE_N = 64                 # points per stroke after resampling, fixed length
                                # so the ML recogniser sees a constant input size
MIN_STROKE_PTS = 4              # anything shorter is a stray tap, discard
SIMPLIFY_TOL_MM = 0.4           # Douglas-Peucker tolerance before fitting

# --- recognition ----------------------------------------------------------- #
CLOSE_TOL_MM = 3.0              # endpoints closer than this = the user meant a closed shape
LINE_RESIDUAL_MM = 0.8          # max mean deviation to accept a straight line
# Calibrated against 325 real mouse-drawn strokes (PK, 2026-09-24), not guessed.
# Real circles fit with a median radial residual of 2.38 mm (p90 4.17). At the
# old 1.2 mm only 18% of them passed and the rest fell through to POLYLINE -
# circle F1 on real input was 0.213. Closed competitors sit far higher (rect
# median 6.05, polyline 7.05), so 3.0 separates them cleanly: 77% of real
# circles kept for 16% of rect/polyline wrongly admitted - the best separation
# in the sweep. Re-derive with scripts/calibrate.py when more data arrives.
CIRCLE_RESIDUAL_MM = 3.5        # max mean radial deviation to accept a circle
LOW_CONFIDENCE = 0.55           # below this the UI flags the shape for redraw

# --- geometry -------------------------------------------------------------- #
ARC_SEGMENTS = 64               # polygon segments per full circle when tessellating
MIN_FEATURE_DEPTH_MM = 0.2
DEFAULT_DEPTH_MM = 10.0
CUT_OVERSHOOT_MM = 0.5          # extend cuts slightly past the body so the boolean
                                # never leaves a zero-thickness coplanar face

# --- printability ---------------------------------------------------------- #
MIN_WALL_MM = 1.2               # thinner than this and most FDM printers cannot do it
MAX_BUILD_MM = (220, 220, 250)  # a common Ender-3 sized build volume
RECT_FILL_MIN = 0.82           # stroke area / min-area-rect area to accept a RECT
SMOOTH_WINDOW = 5               # moving-average window for hand tremor; 1 disables
BRUSH_PX = 2                    # pen width on the canvas; fat strokes round off corners
# Only 63% of PK's real mouse-drawn circles read as closed at 0.06; a circle the
# app calls open becomes an ARC and never gets a circle fit at all. 0.12 lifts
# that without letting genuine arcs (which sweep well under a full turn) through.
CLOSE_FRACTION = 0.12           # gap/length under this also counts as closed


# --- corner detection (ML1 v2, the "star fix") ----------------------------- #
# The old polyline path resampled to 64 points and smoothed with a window of 5,
# which rounds a star's tips off before anything ever looks for them. Corners
# are therefore found on the RAW stroke, before prepare_points() touches it.
# Swept over 7 synthetic shapes (star, L, slot, circle, rect, triangle, hexagon)
# x 4 hand styles x several seeds. Exact on clean and NEAT strokes; degrades at
# MOUSE-level tremor, which is the honest baseline the learned corner model has
# to beat. Re-run scripts/tune_corners.py if the synthesis changes.
CORNER_WINDOW_FRACTION = 0.05   # tangent window, as a fraction of stroke length.
CORNER_WINDOW_MM = 2.5          # lower clamp
CORNER_WINDOW_MAX_MM = 14.0     # upper clamp
CORNER_DENOISE_FRACTION = 0.012 # light pre-smoothing, SCORING ONLY, never fitting
CORNER_DENOISE_MAX_MM = 5.0
CORNER_ANGLE_DEG = 38.0         # absolute floor: below this it is never a corner
CORNER_ANGLE_MAX_DEG = 55.0     # ceiling on the adaptive threshold. Without it a
                                # star fails: its median turning angle is 30 deg
                                # because the shape is mostly corner, so
                                # median + k*MAD lands at 119 and every 64-degree
                                # inner vertex is discarded.
CORNER_MAD_K = 6.0              # how far above its own stroke's noise a point
                                # has to turn. Protects shaky strokes.
CORNER_MIN_SEP_MM = 4.0         # absolute floor on corner separation; the
                                # tangent window usually dominates it
CORNER_SPEED_WEIGHT = 0.35      # how much a pen slowdown adds to the corner
                                # score when timestamps exist. People
                                # decelerate into a deliberate corner and not
                                # into wobble.
SEGMENT_MIN_PTS = 5             # a piece shorter than this cannot be fitted
SEGMENT_LINE_RESIDUAL_MM = 1.0  # piece is a line if it fits one this well
SEGMENT_RESIDUAL_FRACTION = 0.04  # ...or this fraction of its own length, so a
                                # 200 mm edge is not held to a 10 mm edge's
                                # tolerance
SEGMENT_SPLIT_DEPTH = 3         # a piece that fits neither a line nor an arc is
                                # split at its worst point and re-fitted. This is
                                # what recovers a slot: its line-arc junctions are
                                # TANGENT, so they turn through zero degrees and
                                # no corner detector will ever see them.
VERTEX_SNAP_MM = 6.0            # two fitted lines meeting within this distance
                                # of their shared corner get a sharp intersected
                                # vertex instead of a rounded join

# --- multi-stroke contours (sketch contour) -------------------------------- #
STROKE_JOIN_MM = 8.0            # endpoints closer than this may be joined.
                                # Deliberately looser than CLOSE_TOL_MM (3.0),
                                # which is the "did ONE stroke close on itself"
                                # test; people are sloppier between strokes.
STROKE_JOIN_FRACTION = 0.10     # or under this fraction of the total contour
                                # length, so big shapes are not penalised
CONTOUR_IDLE_S = 1.2            # pen up for this long = the shape is finished
SEAM_SMOOTH_MM = 1.5           # arc length each side of a seam to smooth, so a
                                # join does not read as a corner to ML1
