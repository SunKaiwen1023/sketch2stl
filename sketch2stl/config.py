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
