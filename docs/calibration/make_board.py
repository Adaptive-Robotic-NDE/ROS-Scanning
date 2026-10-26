"""The calibration checkerboard, as a vector PDF at exact scale.

9 x 6 INNER CORNERS is what cv2.findChessboardCornersSB is asked for in
calibration.dpi, which means 10 x 7 SQUARES. At the module's default 25 mm
square that is 250 x 175 mm, which does not clear the unprintable margin of a
US Letter sheet. 20 mm gives 200 x 140 mm, which clears it comfortably in
landscape -- so set CAL_BOARD_SQUARE = 0.020 to match this sheet.

THE BORDER IS A MEASURING FEATURE, NOT DECORATION. Its four bars are placed so
their INNER faces lie exactly on the pattern boundary, 200.00 x 140.00 mm
apart. A caliper's inside jaws land on those faces, which is the only crisp
physical edge a checkerboard has -- the outer squares alternate, so half the
nominal boundary is white paper with nothing printed on it.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

MM = 1.0/25.4
COLS_IN, ROWS_IN = 9, 6                 # INNER corners, what OpenCV is told
SQ = 20.0                               # mm
NX, NY = COLS_IN + 1, ROWS_IN + 1       # squares
W, H = NX*SQ, NY*SQ                     # 200 x 140 mm
PAGE_W, PAGE_H = 279.4, 215.9           # US Letter, landscape
BAR = 0.4                               # border bar thickness, mm

x0, y0 = (PAGE_W - W)/2.0, (PAGE_H - H)/2.0 + 14.0

fig = plt.figure(figsize=(PAGE_W*MM, PAGE_H*MM))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, PAGE_W); ax.set_ylim(0, PAGE_H); ax.axis("off")
ax.set_aspect("equal")

# The board. Top-left square black, which is the convention the detector's
# origin assumes; the solve is invariant to it but a human reading the image is not.
for j in range(NY):
    for i in range(NX):
        if (i + j) % 2 == 0:
            ax.add_patch(Rectangle((x0 + i*SQ, y0 + (NY - 1 - j)*SQ), SQ, SQ,
                                   facecolor="black", edgecolor="none"))

# Border bars: inner faces EXACTLY on the pattern boundary.
for rect in ((x0 - BAR, y0 - BAR, W + 2*BAR, BAR),          # below
             (x0 - BAR, y0 + H,   W + 2*BAR, BAR),          # above
             (x0 - BAR, y0,       BAR,       H),            # left
             (x0 + W,   y0,       BAR,       H)):           # right
    ax.add_patch(Rectangle(rect[:2], rect[2], rect[3],
                           facecolor="black", edgecolor="none"))

ax.text(PAGE_W/2.0, y0 + H + 5.0,
        "TARI calibration board   %d x %d inner corners   %.1f mm squares"
        % (COLS_IN, ROWS_IN, SQ),
        ha="center", va="bottom", fontsize=9, family="monospace")

note = (
    "PRINT AT 100%%.  Turn OFF \"fit to page\" / \"shrink to fit\" / \"scale to fit\".\n"
    "\n"
    "THEN MEASURE IT.  1%% of scale error is 1%% of the solved camera standoff,\n"
    "which at 0.55 m is 5.5 mm -- most of the error this procedure exists to remove.\n"
    "Caliper INSIDE jaws against the inner faces of the border bars:\n"
    "     long axis %.2f mm (%d squares)        short axis %.2f mm (%d squares)\n"
    "\n"
    "At the dgpy prompt:   CAL_BOARD_SQUARE = 0.020\n"
    "                      set_board_square(<long axis in METRES>, n=%d)\n"
    "\n"
    "Bond it FLAT before use. The caliper fixes scale; nothing fixes a bubble."
    % (W, NX, H, NY, NX))
ax.text(PAGE_W/2.0, y0 - 6.0, note, ha="center", va="top",
        fontsize=6.6, family="monospace", linespacing=1.45)

fig.savefig("tari_calibration_board_20mm_letter.pdf", format="pdf",
            facecolor="white")
print("tari_calibration_board_20mm_letter.pdf")
print("  page   US Letter landscape %.1f x %.1f mm" % (PAGE_W, PAGE_H))
print("  board  %d x %d inner corners, %.1f mm squares -> %.1f x %.1f mm"
      % (COLS_IN, ROWS_IN, SQ, W, H))
print("  margin %.1f mm left/right, %.1f mm top, %.1f mm bottom"
      % (x0, PAGE_H - (y0 + H), y0))
