#########################################################################
# AdaptiveNDE Project                                                   #
# ROS-Scanning Package                                                  #
#########################################################################
# Copyright 2023-2024 University of Dayton                              #
#                                                                       #
# Redistribution and use in source and binary forms, with or without    #
# modification, are permitted provided that the following conditions    #
# are met:                                                              #
#                                                                       #
# 1. Redistributions of source code must retain the above copyright     #
#    notice, this list of conditions and the following disclaimer.      #
# 2. Redistributions in binary form must reproduce the above copyright  #
#    notice, this list of conditions and the following disclaimer in    #
#    the documentation and/or other materials provided with the         #
#    distribution.                                                      #
# 3. Neither the name of the copyright holder nor the names of its      #
#    contributors may be used to endorse or promote products derived    #
#    from this software without specific prior written permission.      #
#                                                                       #
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS   #
# “AS IS” AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT     #
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR #
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT  #
# HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,#
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT      #
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, #
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY #
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT   #
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE #
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.  #
#########################################################################
# This material is based in part upon work supported by the U.S. Air    #
# Force under Contract FA8650-19-D-5230.                                #
#                                                                       #
# DISTRIBUTION A. Approved for public release: distribution unlimited.  #
#########################################################################
# File:  roi_select_ui.py
# Authors: Tyler Lesthaeghe
# Created: 2026-09-29
# Description:
#   Standalone region-of-interest selection window, run as its own
#   process.
# Derived From:
#
#########################################################################
# Revision Log
#
# 2026-09-29:  Initial Commit
#
#########################################################################
# Development Notes
#
#
#########################################################################
"""The region-of-interest window. A STANDALONE PROCESS -- never imported by dgpy.

    python3 roi_select_ui.py <request.json>

Exit 0 and a written result if the operator accepted, 10 if they cancelled,
anything else is a real failure. roi_select.dpi is the caller; the request and
result formats are documented there and nowhere else, so change both together.

WHY A SEPARATE PROCESS. The scan config has already called
include(dgpy, "Qt.dpi", prefer_pyqt=False) and is running spatialnde2's
OpenSceneGraph viewer on the Qt main thread, and a scan stage runs in a worker
thread with a dgpy context. Opening a second Qt window from there means a
second binding in one interpreter, a second event loop, module censoring
walking Qt objects, and any VTK or GL fault taking the acquisition down with
it. A subprocess costs a fork and two temp files and has none of that --
the same trade mesh_uv_blender.py already makes for Blender.

WHY PyQt5 AND vtkmodules. The cell is Ubuntu 22.04 with ROS 2 natively, where
python3-pyqt5 and python3-vtk9 are packaged and usually already pulled in by
RViz. `import vtk` is NOT used: the top-level shim imports every module VTK was
built with, which on some hosts fails outright on a missing libopenvr_api.so.
Importing only the handful of vtkmodules this file uses sidesteps that
build-options problem entirely.

WHAT THE OPERATOR SEES. A 3D view of the surface shaded by height, a toolbar of
four tools, and Accept/Cancel. See roi_select.dpi's docstring for the key map.
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import roi_select_geom as G

from PyQt5 import QtCore, QtGui, QtWidgets

from vtkmodules.vtkCommonCore import vtkPoints, vtkUnsignedCharArray
from vtkmodules.vtkCommonDataModel import vtkPolyData, vtkCellArray
from vtkmodules.vtkRenderingCore import (
    vtkActor, vtkActor2D, vtkCoordinate, vtkPolyDataMapper,
    vtkPolyDataMapper2D, vtkRenderer)
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleTrackballCamera
from vtkmodules.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray

# Registers the OpenGL rendering factory. Unused by name, needed by import.
import vtkmodules.vtkRenderingOpenGL2            # noqa: F401
import vtkmodules.vtkRenderingFreeType           # noqa: F401

from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

# SELECTION HIGHLIGHT: GREEN, AND THE COLOUR IS NOT A TASTE DECISION.
# The height ramp runs blue -> white -> red, so a red highlight is the same
# colour the ramp already gives the TALLEST thing in the scene -- a tall
# unselected mount block (dimmed red) and a selected plate (bright red) then
# differ only in shade. Green appears nowhere in the ramp at any height, so
# "selected" is legible without reference to anything else in the view.
HIGHLIGHT = np.array([40, 255, 110], dtype=np.float64)
#: How far the unselected surface is knocked back. It keeps its height colour:
#: that shading is how the operator finds the part before selecting it.
DIM = 0.5

TOOLS = ("navigate", "rect", "lasso", "brush")

HELP = """\
<b>Pick the region to scan.</b> Everything not selected is left alone --
no Keyence raster over it, no eddy-current passes over it.

<table cellpadding=3>
<tr><td><b>1</b></td><td>Navigate</td><td>left drag orbits, middle pans, right or wheel zooms</td></tr>
<tr><td><b>2</b></td><td>Rectangle</td><td>drag a box</td></tr>
<tr><td><b>3</b></td><td>Lasso</td><td>drag freehand, or click corners and press Enter / right-click to close</td></tr>
<tr><td><b>4</b></td><td>Brush</td><td>drag to paint; <b>[</b> and <b>]</b> change the radius, or the wheel with Shift held</td></tr>
</table>

<table cellpadding=3>
<tr><td><b>Ctrl</b> + drag</td><td>deselect with whichever tool is active</td></tr>
<tr><td><b>Shift</b> + drag</td><td>add to the selection instead of replacing it</td></tr>
<tr><td><b>Ctrl+A</b> / <b>Ctrl+D</b></td><td>select all / clear</td></tr>
<tr><td><b>Ctrl+I</b></td><td>invert</td></tr>
<tr><td><b>Ctrl+Z</b></td><td>undo</td></tr>
<tr><td><b>T F S I</b></td><td>top, front, side, isometric view</td></tr>
<tr><td><b>R</b></td><td>reset the camera to fit</td></tr>
<tr><td><b>Enter</b></td><td>Accept &nbsp;&nbsp; <b>Esc</b> Cancel</td></tr>
</table>

Middle-drag and right-drag always move the camera, whatever tool is selected,
so you never have to switch back to Navigate just to look at something.
"""


# ==============================================================================

class Style(vtkInteractorStyleTrackballCamera):
    """Left button belongs to the active tool; every other button to the camera.

    Observers rather than overridden On* methods: that is the pattern VTK's
    Python wrapping actually supports for a style subclass, and forwarding to
    the camera is then an explicit call to the parent handler rather than a
    super() that may not be wired up.
    """

    def __init__(self, win):
        self.win = win
        self.AddObserver("LeftButtonPressEvent", self._down)
        self.AddObserver("MouseMoveEvent", self._move)
        self.AddObserver("LeftButtonReleaseEvent", self._up)
        self.AddObserver("RightButtonPressEvent", self._right)
        self.AddObserver("MouseWheelForwardEvent", self._wheel_in)
        self.AddObserver("MouseWheelBackwardEvent", self._wheel_out)

    def _pos(self):
        return self.GetInteractor().GetEventPosition()

    def _mode(self):
        """Ctrl deselects, Shift adds, neither replaces -- unless the toolbar
        has been set to a sticky mode, which then supplies the default."""
        it = self.GetInteractor()
        if it.GetControlKey():
            return "subtract"
        if it.GetShiftKey():
            return "add"
        return self.win.sticky_mode()

    def _down(self, obj, ev):
        if self.win.tool == "navigate":
            self.OnLeftButtonDown()
            return
        self.win.begin_stroke(self._pos(), self._mode())

    def _move(self, obj, ev):
        if self.win.tool == "brush":
            self.win.hover_brush(self._pos())
        if self.win.stroking:
            self.win.extend_stroke(self._pos())
            return
        self.OnMouseMove()

    def _up(self, obj, ev):
        if self.win.stroking:
            self.win.end_stroke(self._pos())
            return
        self.OnLeftButtonUp()

    def _right(self, obj, ev):
        # A right-click closes an open lasso; otherwise it is the camera's.
        if self.win.tool == "lasso" and self.win.polygon:
            self.win.close_polygon()
            return
        self.OnRightButtonDown()

    def _wheel_in(self, obj, ev):
        if self.win.tool == "brush" and self.GetInteractor().GetShiftKey():
            self.win.bump_radius(1.25)
            return
        self.OnMouseWheelForward()

    def _wheel_out(self, obj, ev):
        if self.win.tool == "brush" and self.GetInteractor().GetShiftKey():
            self.win.bump_radius(0.8)
            return
        self.OnMouseWheelBackward()


# ==============================================================================

class RoiWindow(QtWidgets.QMainWindow):

    def __init__(self, req):
        super().__init__()
        self.req = req
        self.accepted = False

        self.verts, self.normals, self.tris = G.read_ply(req["input"])
        self.n = len(self.verts)
        self.base_rgb = G.height_colors(self.verts[:, 2])

        self.mask = G.region_mask(self.verts, req.get("initial"))
        self.undo = []
        self.tool = "rect"
        self.stroking = False
        self.stroke = []
        self.polygon = []
        self.mode = "replace"
        self.radius = float(req.get("brush_radius", 40.0))
        self.front_only = bool(req.get("front_only", True))

        # THE TWO NUMBERS THAT DECIDE WHETHER A SELECTION IS SCANNABLE, both
        # supplied by the caller because neither is a property of the surface on
        # screen: the taught work area the region is clipped to, and the line
        # spacing it has to be wide enough for. None for either leaves that
        # check off -- no work area means no clip, and no spacing falls back to
        # roi_select_geom's own default.
        wa = req.get("work_area")
        self.work_area = None if wa is None else [float(v) for v in wa]
        ls = req.get("line_spacing")
        self.line_spacing = None if ls is None else float(ls)

        # Projected screen coordinates, recomputed only when the camera moves.
        self._xy = None
        self._depth = None

        self.setWindowTitle(req.get("title") or "Select the region to scan")
        self._build()
        self._build_scene()
        self.refresh(reset_camera=True)

    # -- chrome ------------------------------------------------------------

    def _build(self):
        self.frame = QtWidgets.QFrame()
        self.setCentralWidget(self.frame)
        lay = QtWidgets.QVBoxLayout(self.frame)
        lay.setContentsMargins(0, 0, 0, 0)

        self.vtkw = QVTKRenderWindowInteractor(self.frame)
        lay.addWidget(self.vtkw, 1)

        bar = self.addToolBar("tools")
        bar.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        bar.setMovable(False)

        self.tool_group = QtWidgets.QActionGroup(self)
        for key, label, tip in (
                ("navigate", "&Navigate", "Orbit, pan and zoom (1)"),
                ("rect", "&Rectangle", "Drag a box (2)"),
                ("lasso", "&Lasso", "Drag freehand, or click corners and "
                                    "press Enter to close (3)"),
                ("brush", "&Brush", "Drag to paint; [ and ] resize (4)")):
            act = QtWidgets.QAction(label, self)
            act.setCheckable(True)
            act.setToolTip(tip)
            act.setData(key)
            act.triggered.connect(lambda _c, k=key: self.set_tool(k))
            self.tool_group.addAction(act)
            bar.addAction(act)
            if key == self.tool:
                act.setChecked(True)
        bar.addSeparator()

        bar.addWidget(QtWidgets.QLabel(" Drag does: "))
        self.mode_box = QtWidgets.QComboBox()
        self.mode_box.addItems(["select", "deselect", "add to selection"])
        self.mode_box.setToolTip("What a plain drag does. Ctrl always "
                                 "deselects and Shift always adds, whatever "
                                 "this says.")
        self.mode_box.currentIndexChanged.connect(self._mode_changed)
        bar.addWidget(self.mode_box)

        bar.addWidget(QtWidgets.QLabel("  Brush px: "))
        self.radius_spin = QtWidgets.QSpinBox()
        self.radius_spin.setRange(3, 600)
        self.radius_spin.setValue(int(self.radius))
        self.radius_spin.valueChanged.connect(self._radius_changed)
        bar.addWidget(self.radius_spin)

        self.front_box = QtWidgets.QCheckBox("near surface only")
        self.front_box.setChecked(self.front_only)
        self.front_box.setToolTip(
            "Take only what is nearest the camera in each patch of screen, so "
            "a shape drawn over the part does not also grab its underside or "
            "the bench straight behind it.")
        self.front_box.stateChanged.connect(self._front_changed)
        bar.addWidget(self.front_box)

        bar.addSeparator()
        for label, slot, tip in (
                ("All", self.select_all, "Select everything (Ctrl+A)"),
                ("Clear", self.select_none, "Select nothing (Ctrl+D)"),
                ("Invert", self.invert, "Invert the selection (Ctrl+I)"),
                ("Undo", self.undo_last, "Undo the last change (Ctrl+Z)")):
            a = QtWidgets.QAction(label, self)
            a.setToolTip(tip)
            a.triggered.connect(slot)
            bar.addAction(a)

        bar.addSeparator()
        bar.addWidget(QtWidgets.QLabel(" View: "))
        for label, key in (("Top", "T"), ("Front", "F"), ("Side", "S"),
                           ("Iso", "I")):
            a = QtWidgets.QAction(label, self)
            a.setToolTip("%s view (%s)" % (label, key))
            a.triggered.connect(lambda _c, k=label.lower(): self.set_view(k))
            bar.addAction(a)
        a = QtWidgets.QAction("Fit", self)
        a.setToolTip("Reset the camera to fit the surface (R)")
        a.triggered.connect(lambda: self.set_view(None))
        bar.addAction(a)

        spacer = QtWidgets.QWidget()
        spacer.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                             QtWidgets.QSizePolicy.Preferred)
        bar.addWidget(spacer)
        help_act = QtWidgets.QAction("&Help", self)
        help_act.triggered.connect(self.show_help)
        bar.addAction(help_act)

        # Accept / Cancel, big and at the bottom where a decision belongs.
        row = QtWidgets.QHBoxLayout()
        row.setContentsMargins(8, 4, 8, 8)
        self.status = QtWidgets.QLabel("")
        self.status.setTextFormat(QtCore.Qt.PlainText)
        row.addWidget(self.status, 1)

        # NEITHER OF THESE MAY BE ACTIVATED BY A KEYPRESS. This window appears
        # while the operator may be typing somewhere else, and it takes focus: a
        # Space that arrives with Accept focused commits a region nobody drew and
        # starts the Keyence raster over it, recoverable only by killing the run.
        #
        # Three independent guards stand against that: the window opens with
        # nothing selected (see __init__), a stray Accept lands on a confirmation
        # (do_accept), and neither button can hold focus at all:
        #
        #   NoFocus       Space and Enter go nowhere near them. This is the one
        #                 that actually closes the hole -- the others are depth.
        #   autoDefault   off, so Enter anywhere does not reach them either.
        #                 setDefault(True) implies autoDefault, which would make
        #                 Accept the target of a bare Return.
        #
        # Enter and Esc still work, deliberately -- an operator who wants the
        # keyboard should have it -- but as explicit window-level shortcuts
        # (below) that run the same confirmation path as a click, not as button
        # activations that bypass it.
        self.cancel_btn = QtWidgets.QPushButton("Cancel  (Esc)")
        self.cancel_btn.setMinimumWidth(170)
        self.cancel_btn.setAutoDefault(False)
        self.cancel_btn.setDefault(False)
        self.cancel_btn.setFocusPolicy(QtCore.Qt.NoFocus)
        self.cancel_btn.clicked.connect(self.do_cancel)
        row.addWidget(self.cancel_btn)

        self.accept_btn = QtWidgets.QPushButton("Accept  (Enter)")
        self.accept_btn.setMinimumWidth(190)
        self.accept_btn.setAutoDefault(False)
        self.accept_btn.setDefault(False)
        self.accept_btn.setFocusPolicy(QtCore.Qt.NoFocus)
        font = self.accept_btn.font()
        font.setBold(True)
        self.accept_btn.setFont(font)
        self.accept_btn.clicked.connect(self.do_accept)
        row.addWidget(self.accept_btn)
        lay.addLayout(row)

        for seq, slot in (
                ("1", lambda: self.set_tool("navigate")),
                ("2", lambda: self.set_tool("rect")),
                ("3", lambda: self.set_tool("lasso")),
                ("4", lambda: self.set_tool("brush")),
                ("Ctrl+A", self.select_all),
                ("Ctrl+D", self.select_none),
                ("Ctrl+I", self.invert),
                ("Ctrl+Z", self.undo_last),
                ("[", lambda: self.bump_radius(0.8)),
                ("]", lambda: self.bump_radius(1.25)),
                ("T", lambda: self.set_view("top")),
                ("F", lambda: self.set_view("front")),
                ("S", lambda: self.set_view("side")),
                ("I", lambda: self.set_view("iso")),
                ("R", lambda: self.set_view(None)),
                ("Return", self.do_accept),
                ("Enter", self.do_accept),
                ("Esc", self.do_cancel),
                ("F1", self.show_help)):
            QtWidgets.QShortcut(QtGui.QKeySequence(seq), self, slot)

        self.resize(1180, 840)

    # -- the scene ---------------------------------------------------------

    def _build_scene(self):
        self.poly = vtkPolyData()
        pts = vtkPoints()
        pts.SetData(numpy_to_vtk(np.ascontiguousarray(self.verts), deep=1))
        self.poly.SetPoints(pts)

        if self.tris is not None and len(self.tris):
            cells = np.empty((len(self.tris), 4), dtype=np.int64)
            cells[:, 0] = 3
            cells[:, 1:] = self.tris
            ca = vtkCellArray()
            ca.SetCells(len(self.tris),
                        numpy_to_vtkIdTypeArray(cells.ravel(), deep=1))
            self.poly.SetPolys(ca)
        else:
            idx = np.arange(self.n, dtype=np.int64)
            cells = np.empty((self.n, 2), dtype=np.int64)
            cells[:, 0] = 1
            cells[:, 1] = idx
            ca = vtkCellArray()
            ca.SetCells(self.n, numpy_to_vtkIdTypeArray(cells.ravel(), deep=1))
            self.poly.SetVerts(ca)

        # One persistent colour buffer, mutated in place. Rebuilding a vtk array
        # of 100 k x 3 -- an ordinary surface here -- on every mouse move is the
        # difference between a brush that paints and a brush that stutters.
        self._rgb = np.ascontiguousarray(self.base_rgb.copy())
        self.colors = numpy_to_vtk(self._rgb, deep=0,
                                   array_type=vtkUnsignedCharArray().GetDataType())
        self.colors.SetName("rgb")
        self.poly.GetPointData().SetScalars(self.colors)

        mapper = vtkPolyDataMapper()
        mapper.SetInputData(self.poly)
        mapper.SetScalarModeToUsePointData()
        mapper.SetColorModeToDirectScalars()
        self.actor = vtkActor()
        self.actor.SetMapper(mapper)
        prop = self.actor.GetProperty()
        prop.SetPointSize(3.0 if self.tris is None else 1.0)
        prop.SetAmbient(0.35)
        prop.SetDiffuse(0.75)
        prop.SetSpecular(0.0)          # no specular: it reads as false relief

        self.ren = vtkRenderer()
        self.ren.SetBackground(0.14, 0.15, 0.18)
        self.ren.AddActor(self.actor)
        self.ren.AddActor(self._overlay())

        rw = self.vtkw.GetRenderWindow()
        rw.AddRenderer(self.ren)
        # Software GL over VNC: multisampling is the first thing to fall over,
        # and it buys nothing at this point size.
        rw.SetMultiSamples(0)

        self.iren = rw.GetInteractor()
        self.style = Style(self)
        self.iren.SetInteractorStyle(self.style)
        self.ren.GetActiveCamera().AddObserver("ModifiedEvent",
                                               self._camera_moved)

    def _overlay(self):
        """The rubber band / lasso outline / brush circle, in display coords."""
        self.ov_poly = vtkPolyData()
        self.ov_poly.SetPoints(vtkPoints())
        self.ov_poly.SetLines(vtkCellArray())
        coord = vtkCoordinate()
        coord.SetCoordinateSystemToDisplay()
        m = vtkPolyDataMapper2D()
        m.SetInputData(self.ov_poly)
        m.SetTransformCoordinate(coord)
        self.ov_actor = vtkActor2D()
        self.ov_actor.SetMapper(m)
        self.ov_actor.GetProperty().SetColor(1.0, 0.95, 0.2)
        self.ov_actor.GetProperty().SetLineWidth(1.6)
        return self.ov_actor

    def start(self):
        self.show()
        self.iren.Initialize()
        self.set_view("top")

    # -- projection cache --------------------------------------------------

    def _camera_moved(self, *a):
        self._xy = None

    def screen(self):
        """(xy, depth) for every vertex under the camera as it is right now."""
        if self._xy is None:
            cam = self.ren.GetActiveCamera()
            w, h = self.vtkw.GetRenderWindow().GetSize()
            aspect = (float(w)/float(h)) if h else 1.0
            m = cam.GetCompositeProjectionTransformMatrix(aspect, -1.0, 1.0)
            mat = np.array([[m.GetElement(i, j) for j in range(4)]
                            for i in range(4)], dtype=np.float64)
            self._xy, self._depth = G.project(self.verts, mat, w, h)
        return self._xy, self._depth

    # -- the tools ---------------------------------------------------------

    def sticky_mode(self):
        return ("replace", "subtract", "add")[self.mode_box.currentIndex()]

    def set_tool(self, key):
        self.abort_stroke()
        self.tool = key
        for act in self.tool_group.actions():
            act.setChecked(act.data() == key)
        self.vtkw.setCursor(QtGui.QCursor(
            QtCore.Qt.OpenHandCursor if key == "navigate"
            else QtCore.Qt.CrossCursor))
        self.draw_overlay([])
        self.refresh()

    def begin_stroke(self, pos, mode):
        self.mode = mode
        if self.tool == "lasso":
            # A click adds a corner; a drag from here becomes freehand. Both
            # end up in self.polygon, so close_polygon() is the one exit.
            self.polygon.append((float(pos[0]), float(pos[1])))
            self.stroking = True
            self.stroke = [(float(pos[0]), float(pos[1]))]
            self._lasso_dragged = False
            self.draw_overlay(self.polygon, closed=False)
            # Renders too: a lasso can be built by CLICKING corners rather than
            # dragging, and without this the outline between placed corners is
            # only drawn when something else happens to render.
            self.vtkw.GetRenderWindow().Render()
            return
        self.stroking = True
        self.stroke = [(float(pos[0]), float(pos[1]))]
        if self.tool == "brush":
            self.push_undo()
            self.paint(self.stroke)

    def extend_stroke(self, pos):
        """Grow the stroke AND SHOW IT.

        draw_overlay() only rewrites the overlay's vtkPolyData; nothing appears
        until the render window is asked to draw. Without the explicit Render()
        for rect and lasso the rubber band and the lasso outline are built
        correctly and never shown, and the operator draws blind. Brush does not
        need one only because it paints as it goes and paint() -> refresh()
        renders.
        """
        p = (float(pos[0]), float(pos[1]))
        if self.tool == "rect":
            a = self.stroke[0]
            self.draw_overlay([a, (p[0], a[1]), p, (a[0], p[1])], closed=True)
            self.vtkw.GetRenderWindow().Render()
        elif self.tool == "lasso":
            self._lasso_dragged = True
            if (abs(p[0] - self.polygon[-1][0]) > 2 or
                    abs(p[1] - self.polygon[-1][1]) > 2):
                self.polygon.append(p)
            self.draw_overlay(self.polygon, closed=False)
            self.vtkw.GetRenderWindow().Render()
        elif self.tool == "brush":
            self.stroke.append(p)
            self.paint(self.stroke[-2:])
            self.draw_overlay(self._circle(p), closed=True)

    def end_stroke(self, pos):
        p = (float(pos[0]), float(pos[1]))
        self.stroking = False
        if self.tool == "rect":
            a = self.stroke[0]
            self.push_undo()
            xy, depth = self.screen()
            shape = G.mask_from_box(xy, a[0], a[1], p[0], p[1])
            self.apply(shape)
            self.draw_overlay([])
        elif self.tool == "lasso":
            if self._lasso_dragged:
                self.close_polygon()
            # else: it was a click, so the corner stays and we wait for more.
        elif self.tool == "brush":
            self.draw_overlay(self._circle(p), closed=True)
        self.refresh()

    def close_polygon(self):
        poly = list(self.polygon)
        self.polygon = []
        self.stroking = False
        self.draw_overlay([])
        if len(poly) >= 3:
            self.push_undo()
            xy, _ = self.screen()
            self.apply(G.mask_from_polygon(xy, poly))
        self.refresh()

    def abort_stroke(self):
        self.stroking = False
        self.polygon = []
        self.stroke = []
        self.draw_overlay([])

    def paint(self, segment):
        xy, _ = self.screen()
        self.apply(G.mask_from_stroke(xy, segment, self.radius),
                   mode=("subtract" if self.mode == "subtract" else "add"))
        self.refresh(status_only=False)

    def apply(self, shape, mode=None):
        """Fold a tool's shape in, optionally through the near-surface filter."""
        if self.front_only and shape.any():
            xy, depth = self.screen()
            shape = G.front_surface_mask(xy, depth, shape)
        self.mask = G.combine(self.mask, shape, mode or self.mode)

    def hover_brush(self, pos):
        if not self.stroking:
            self.draw_overlay(self._circle((float(pos[0]), float(pos[1]))),
                              closed=True)
            self.vtkw.GetRenderWindow().Render()

    def _circle(self, centre, n=48):
        t = np.linspace(0.0, 2.0*np.pi, n, endpoint=False)
        return list(zip(centre[0] + self.radius*np.cos(t),
                        centre[1] + self.radius*np.sin(t)))

    def bump_radius(self, factor):
        self.radius_spin.setValue(
            int(max(3, min(600, round(self.radius*factor)))))

    # -- whole-selection operations ---------------------------------------

    def push_undo(self):
        self.undo.append(self.mask.copy())
        del self.undo[:-40]

    def select_all(self):
        self.push_undo()
        self.mask = np.ones(self.n, dtype=bool)
        self.refresh()

    def select_none(self):
        self.push_undo()
        self.mask = np.zeros(self.n, dtype=bool)
        self.refresh()

    def invert(self):
        self.push_undo()
        self.mask = ~self.mask
        self.refresh()

    def undo_last(self):
        if self.undo:
            self.mask = self.undo.pop()
            self.refresh()

    # -- drawing -----------------------------------------------------------

    def draw_overlay(self, points, closed=False):
        pts = vtkPoints()
        lines = vtkCellArray()
        if len(points) >= 2:
            for x, y in points:
                pts.InsertNextPoint(float(x), float(y), 0.0)
            n = len(points)
            lines.InsertNextCell(n + (1 if closed else 0))
            for i in range(n):
                lines.InsertCellPoint(i)
            if closed:
                lines.InsertCellPoint(0)
        self.ov_poly.SetPoints(pts)
        self.ov_poly.SetLines(lines)
        self.ov_poly.Modified()

    def refresh(self, reset_camera=False, status_only=False):
        sel = self.mask
        self._rgb[:] = (self.base_rgb*DIM).astype(np.uint8)
        if sel.any():
            # Blend rather than replace, so relief stays readable under the
            # highlight -- a flat red patch hides the very step the operator
            # is trying to keep out of the selection.
            self._rgb[sel] = np.clip(
                0.45*self.base_rgb[sel] + 0.55*HIGHLIGHT, 0, 255).astype(np.uint8)
        self.colors.Modified()
        self.poly.Modified()

        n = int(sel.sum())
        if n:
            x0, y0, x1, y1 = G.bounding_region(self.verts, sel)
            z = self.verts[sel][:, 2]
            region, why = self.scan_region()
            if region is None:
                # SHOWN WHILE THEY DRAW, not when they press Accept. The
                # selection is still under the mouse at this point, so the
                # refusal is one more drag to fix.
                txt = ("%d of %d points (%.1f%%)   THIS CANNOT BE SCANNED: %s"
                       % (n, self.n, 100.0*n/self.n, why))
            else:
                txt = ("%d of %d points (%.1f%%)   scan region %.1f x %.1f mm "
                       "at x %.3f..%.3f y %.3f..%.3f   (points span "
                       "%.1f x %.1f mm, z %.3f..%.3f m)"
                       % (n, self.n, 100.0*n/self.n,
                          (region[2] - region[0])*1e3,
                          (region[3] - region[1])*1e3,
                          region[0], region[2], region[1], region[3],
                          (x1 - x0)*1e3, (y1 - y0)*1e3, z.min(), z.max()))
        else:
            txt = ("nothing selected -- draw a region with Rectangle, Lasso or "
                   "Brush. Accepting with nothing selected cancels the scan.")
        self.status.setText(txt)
        # Accept stays ENABLED with an empty selection, so do_accept() can ask
        # whether that means "cancel the scan". A disabled button answers the
        # operator's press with silence.
        self.accept_btn.setEnabled(True)

        if reset_camera:
            self.ren.ResetCamera()
        if not status_only:
            self.vtkw.GetRenderWindow().Render()

    def set_view(self, which):
        cam = self.ren.GetActiveCamera()
        if which is not None:
            cam.SetFocalPoint(0.0, 0.0, 0.0)
            cam.SetPosition({"top": (0.0, 0.0, 1.0),
                             "front": (0.0, -1.0, 0.0),
                             "side": (1.0, 0.0, 0.0),
                             "iso": (0.6, -0.8, 0.9)}[which])
            cam.SetViewUp((0.0, 1.0, 0.0) if which != "front" and which != "side"
                          else (0.0, 0.0, 1.0))
        self.ren.ResetCamera()
        self._xy = None
        self.vtkw.GetRenderWindow().Render()

    def _mode_changed(self, _i):
        self.refresh(status_only=True)

    def _radius_changed(self, v):
        self.radius = float(v)

    def _front_changed(self, _s):
        self.front_only = self.front_box.isChecked()

    def show_help(self):
        QtWidgets.QMessageBox.information(self, "Selecting a region", HELP)

    # -- the decision ------------------------------------------------------

    def scan_region(self):
        """(the region that would be reported, None), or (None, why it is not).

        Cheap -- a bounding box and some arithmetic, no rasterising -- so the
        status bar can ask on every selection change.
        """
        if not self.mask.any():
            return None, "nothing is selected"
        try:
            return G.selection_region(self.verts, self.mask,
                                      work_area=self.work_area,
                                      line_spacing=self.line_spacing), None
        except G.SelectionRefused as exc:
            return None, str(exc)

    def do_accept(self):
        """Accept the selection, or ask what an empty one means.

        ACCEPT WITH NOTHING SELECTED IS A QUESTION, NOT A REFUSAL. Refusing it
        with a disabled button is the wrong shape for two reasons: it makes
        "I have decided not to scan this" unsayable from the button the operator
        is already looking at -- Esc is the only way out, and it is not
        discoverable -- and a disabled button answers a press with silence.

        So it asks, with the SAFE option as the default: a stray activation lands
        on "Go back" and nothing is lost.
        """
        if not self.mask.any():
            box = QtWidgets.QMessageBox(self)
            box.setIcon(QtWidgets.QMessageBox.Question)
            box.setWindowTitle("Nothing selected")
            box.setText("No region is selected.")
            box.setInformativeText(
                "Accepting with nothing selected CANCELS THE SCAN -- there is "
                "no region to measure, so no Keyence raster and no eddy-current "
                "toolpath will be planned.\n\n"
                "To scan part of this surface, go back and draw a region with "
                "the Rectangle, Lasso or Brush tool.")
            cancel_scan = box.addButton("Cancel the scan",
                                        QtWidgets.QMessageBox.DestructiveRole)
            go_back = box.addButton("Go back and select",
                                    QtWidgets.QMessageBox.RejectRole)
            # The safe one is the default AND the escape, so every accidental
            # dismissal -- Enter, Esc, the window's close button -- goes back.
            box.setDefaultButton(go_back)
            box.setEscapeButton(go_back)
            box.exec()
            if box.clickedButton() is cancel_scan:
                self.accepted = False
                self.close()
            return

        # A SELECTION TOO SMALL TO SCAN IS REFUSED HERE, with the window still
        # open and the selection still on screen. The alternative is what used to
        # happen: it is accepted, the arm flies, and the reconstruction stops the
        # run on a rectangle the operator can no longer see or change.
        region, why = self.scan_region()
        if region is None:
            box = QtWidgets.QMessageBox(self)
            box.setIcon(QtWidgets.QMessageBox.Warning)
            box.setWindowTitle("The region cannot be scanned")
            box.setText(why)
            box.setInformativeText(
                "Nothing has been lost -- the selection is still there. Enlarge "
                "it, or press Cancel to stop the scan.")
            box.addButton("Go back and adjust", QtWidgets.QMessageBox.AcceptRole)
            box.exec()
            return

        self.accepted = True
        self.close()

    def do_cancel(self):
        if self.polygon:            # Esc first abandons an unfinished lasso
            self.abort_stroke()
            self.refresh()
            return
        self.accepted = False
        self.close()


# ==============================================================================

def write_result(win, req):
    mask = win.mask

    # THE DRAWN SHAPE, in world x-y, AND ITS PER-CELL HEIGHTS, so it can be
    # applied to a DIFFERENT mesh. The per-vertex mask below is exact but is
    # indexed over this surface alone: the ROI is drawn on the Azure rough mesh
    # and the mesh it has to crop is the Keyence one, with a different vertex
    # count and ordering, so the indices cannot line up and never will. A
    # footprint is in metres, and its height map is what lets the crop stay on
    # the surface the operator picked instead of on its shadow.
    #
    # Built FIRST because the region below is widened to cover it: the region
    # says what to measure and the footprint says what to plan on, and measuring
    # less than will be planned on is not a saving.
    fp = None
    shape = {}
    try:
        fp = G.footprint(win.verts, mask, tris=win.tris)
        shape["footprint"] = G.footprint_encode(fp)
        shape["footprint_cells"] = int(fp["grid"].sum())
        shape["footprint_area_cm2"] = G.footprint_area_m2(fp)*1e4
        shape["footprint_bounds"] = list(G.footprint_bounds(fp))
        shape["footprint_described"] = G.describe_footprint(fp)
        shape["footprint_has_heights"] = bool(G.footprint_has_heights(fp))
    except Exception as exc:
        # Never lose a selection the operator has already made over this: the
        # region and the mask still describe it, and a missing footprint makes
        # the crop fall back to the region rectangle with a note.
        fp = None
        shape["footprint_error"] = "%s: %s" % (type(exc).__name__, exc)
        sys.stderr.write("ROISELECT could not build a footprint (%s)\n" % (exc,))

    # THE RECTANGLE THE PIPELINE SCANS: whole selection cells, clipped to the
    # taught work area. NOT the raw span of the selected vertices -- on a 6 mm
    # mesh that span is quantised far coarser than the 0.3 mm line spacing and
    # has come back as zero on one axis for a region with obvious area. See
    # roi_select_geom.selection_region(). do_accept() has already refused
    # anything this would reject, so it does not raise here.
    drawn = G.bounding_region(win.verts, mask)
    region = G.selection_region(win.verts, mask, footprint=fp,
                                work_area=win.work_area,
                                line_spacing=win.line_spacing)
    out = {"region": list(region),
           # What the selected VERTICES span, before that was rounded out to
           # whole cells and clipped. Provenance: it is how far a reported region
           # sits from the sampling it was measured on.
           "region_drawn": list(drawn),
           "region_described": G.describe_selection_region(region, drawn),
           # PROVENANCE ONLY. bounds3d says where the surface the operator was
           # LOOKING AT sat in z, on the sensor that measured it. It is not a
           # crop: an ROI's z floor taken from the Azure rough mesh can sit above
           # the top of the Keyence mesh of the same part, so cropping the
           # Keyence mesh with it keeps zero faces. A GLOBAL z range does not
           # travel between sensors. What travels is `footprint` below -- the
           # drawn shape in world x-y, which is what an operator looking straight
           # down actually chose, TOGETHER WITH the height of that surface in
           # each of its cells. That is how z reaches the crop as a band that
           # follows the surface rather than as a slab.
           "bounds3d": list(G.bounding_box3d(win.verts, mask)),
           "bounds3d_is_provenance": True,
           "n_selected": int(mask.sum()),
           "n_total": int(win.n),
           "source": req["input"],
           "has_faces": win.tris is not None,
           "front_only": bool(win.front_only)}
    out.update(shape)

    np.save(req["mask"], mask)
    out["mask_file"] = req["mask"]

    sv, st, sn, _ = G.submesh(win.verts, win.tris, mask, normals=win.normals)
    G.write_ply(req["submesh"], sv, st, sn)
    out["submesh"] = req["submesh"]
    out["submesh_vertices"] = int(len(sv))
    out["submesh_triangles"] = 0 if st is None else int(len(st))
    if win.tris is not None:
        out["triangles_total"] = int(len(win.tris))

    with open(req["output"], "w") as fh:
        json.dump(out, fh, indent=1)
    return out


def main(argv):
    if len(argv) != 2:
        sys.stderr.write("usage: roi_select_ui.py <request.json | mesh.ply>\n")
        return 2

    if argv[1].lower().endswith(".json"):
        with open(argv[1]) as fh:
            req = json.load(fh)
    else:
        # A PLY straight off the command line, so the window can be opened on a
        # real scan on the production system without dgpy and without
        # hand-writing a request
        # -- which is how you tell "the selector is broken" from "the scan is
        # broken" in one command.
        import tempfile
        src = os.path.abspath(argv[1])
        out = tempfile.mkdtemp(prefix="roi_select_cli_")
        req = {"input": src, "title": os.path.basename(src), "initial": None,
               "output": os.path.join(out, "result.json"),
               "mask": os.path.join(out, "mask.npy"),
               "submesh": os.path.join(out, "roi_submesh.ply"),
               "brush_radius": 40.0, "front_only": True}
        sys.stderr.write("ROISELECT results will be written to %s\n" % (out,))

    app = QtWidgets.QApplication(sys.argv[:1])
    app.setApplicationName("ROI select")
    win = RoiWindow(req)
    win.start()
    app.exec_()

    # QVTKRenderWindowInteractor holds a render window that must be finalised
    # before the process exits, or a software GL stack over VNC can hang on
    # teardown instead of returning an exit code to the caller.
    try:
        win.vtkw.GetRenderWindow().Finalize()
    except Exception:
        pass

    if not win.accepted:
        sys.stderr.write("ROISELECT cancelled by the operator\n")
        return 10
    out = write_result(win, req)
    sys.stderr.write("ROISELECT %d of %d points, region %.4f %.4f %.4f %.4f\n"
                     % ((out["n_selected"], out["n_total"]) +
                        tuple(out["region"])))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
