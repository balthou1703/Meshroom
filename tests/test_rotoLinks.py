"""Roto links: shapes (and the whole Roto) warped by all the tracks of a Tracker or some of them (meshroom/live/roto.py)."""

import json
import math
import os

import numpy as np
import pytest

import meshroom
from meshroom.core import pluginManager
from meshroom.core.graph import Graph
from meshroom.live import Evaluator, buildPlan, roto

_COMPOSITING_FOLDER = os.path.join(os.path.dirname(meshroom.__file__), "nodes", "compositing")
if not pluginManager.isNodeDescRegistered("Grade"):
    pluginManager.addPluginFromBuiltInFolder("compositing", _COMPOSITING_FOLDER)


def rotatedTracks(angle=30.0, scale=1.5, shift=(10.0, -5.0), center=(100.0, 100.0)):
    """ 4 tracks moving from frame 1 (a square around `center`) to frame 2 (rotated, scaled and shifted). """
    c, s = math.cos(math.radians(angle)) * scale, math.sin(math.radians(angle)) * scale
    tracks = []
    for i, (dx, dy) in enumerate([(-20, -20), (20, -20), (20, 20), (-20, 20)]):
        moved = (center[0] + c * dx - s * dy + shift[0], center[1] + s * dx + c * dy + shift[1])
        tracks.append({"name": f"t{i}", "enabled": True,
                       "keys": {"1": [center[0] + dx, center[1] + dy], "2": list(moved)}})
    # A track going elsewhere (e.g. on another object)
    tracks.append({"name": "other", "enabled": True, "keys": {"1": [0.0, 0.0], "2": [50.0, 0.0]}})
    return tracks


SHAPE = {"name": "s", "points": [{"x": 100.0, "y": 120.0, "inX": 0, "inY": 0, "outX": 5.0, "outY": 0.0},
                                 {"x": 120.0, "y": 100.0}]}


def expected(x, y, angle=30.0, scale=1.5, shift=(10.0, -5.0), center=(100.0, 100.0)):
    c, s = math.cos(math.radians(angle)) * scale, math.sin(math.radians(angle)) * scale
    dx, dy = x - center[0], y - center[1]
    return center[0] + c * dx - s * dy + shift[0], center[1] + s * dx + c * dy + shift[1]


def test_someTracksWarpTheShape():
    tracks = {"tracker": rotatedTracks()}
    shape = dict(SHAPE, links=[{"input": "tracker", "tracks": ["t0", "t1", "t2", "t3"],
                                "transform": "translate rotate scale", "point": -1, "frame": 1}])
    points = roto.shapeAtFrame(shape, 2, tracks)["points"]
    assert (points[0]["x"], points[0]["y"]) == pytest.approx(expected(100.0, 120.0), abs=1e-6)
    # The tangents turn and scale with the shape
    c, s = math.cos(math.radians(30)) * 1.5, math.sin(math.radians(30)) * 1.5
    assert (points[0]["outX"], points[0]["outY"]) == pytest.approx((5 * c, 5 * s), abs=1e-6)
    # At the link frame, the shape is where it was drawn
    assert roto.shapeAtFrame(shape, 1, tracks)["points"][0]["x"] == pytest.approx(100.0)


def test_allTheTracksOfATracker():
    tracks = {"tracker": rotatedTracks()[:4]}
    shape = dict(SHAPE, links=[{"input": "tracker", "tracks": [], "transform": "perspective", "frame": 1}])
    points = roto.shapeAtFrame(shape, 2, tracks)["points"]
    assert (points[1]["x"], points[1]["y"]) == pytest.approx(expected(120.0, 100.0), abs=1e-6)


def test_formerSingleTrackLinksStillTranslate():
    tracks = {"tracker": rotatedTracks()}
    shape = dict(SHAPE, links=[{"input": "tracker", "track": "other", "point": -1, "frame": 1}])
    points = roto.shapeAtFrame(shape, 2, tracks)["points"]
    assert (points[0]["x"], points[0]["y"]) == pytest.approx((150.0, 120.0))


def test_tooFewTracksForTheTransform():
    tracks = {"tracker": rotatedTracks()}
    # Perspective needs 4 tracks: with one, its motion
    shape = dict(SHAPE, links=[{"input": "tracker", "tracks": ["other"], "transform": "perspective", "frame": 1}])
    assert roto.shapeAtFrame(shape, 2, tracks)["points"][1]["x"] == pytest.approx(170.0)


def test_rotoLinksWarpEveryShapeAfterTheirOwnLinks():
    tracks = {"tracker": rotatedTracks()}
    globalLinks = [{"input": "tracker", "tracks": ["t0", "t1", "t2", "t3"], "transform": "translate rotate scale",
                    "frame": 1}]
    shape = dict(SHAPE, links=[{"input": "tracker", "tracks": ["other"], "point": 1, "frame": 1}])
    points = roto.shapeAtFrame(shape, 2, tracks, globalLinks)["points"]
    # Point 0: the Roto links only; point 1: its own link (+50 in x), then the Roto links
    assert (points[0]["x"], points[0]["y"]) == pytest.approx(expected(100.0, 120.0), abs=1e-6)
    assert (points[1]["x"], points[1]["y"]) == pytest.approx(expected(170.0, 100.0), abs=1e-6)


def test_relinkKeepsTheShapeWhereItIs():
    tracks = {"tracker": rotatedTracks()}
    shape = dict(SHAPE, keys=[{"frame": 1, "points": SHAPE["points"]}, {"frame": 3, "points": SHAPE["points"]}])
    links = [{"input": "tracker", "tracks": [], "transform": "translate rotate scale", "frame": 1}]
    before = roto.shapeAtFrame(shape, 2, tracks)["points"]
    relinked = roto.relinkShape(shape, 2, tracks, links=links)
    after = roto.shapeAtFrame(relinked, 2, tracks)["points"]
    for a, b in zip(before, after):
        assert (a["x"], a["y"], a.get("outX", 0)) == pytest.approx((b["x"], b["y"], b["outX"]), abs=1e-6)
    # The same with the links of the Roto
    globalLinks = [{"input": "tracker", "tracks": ["other"], "frame": 1}]
    relinked = roto.relinkShape(shape, 2, tracks, [], newGlobalLinks=globalLinks)
    after = roto.shapeAtFrame(relinked, 2, tracks, globalLinks)["points"]
    assert (after[0]["x"], after[0]["y"]) == pytest.approx((before[0]["x"], before[0]["y"]), abs=1e-6)


def test_rotoNodeRendersWithItsLinks():
    graph = Graph("rotoLinks")
    tracker = graph.addNewNode("Tracker")
    constant = graph.addNewNode("Constant", width=200, height=200)
    rotoNode = graph.addNewNode("Roto")
    graph.addEdge(constant.output, tracker.input)
    graph.addEdge(constant.output, rotoNode.input)
    graph.addEdge(tracker.output, rotoNode.tracker)
    tracker.tracks.value = json.dumps(rotatedTracks())
    square = {"name": "sq", "points": [{"x": x, "y": y} for x, y in ((90, 90), (110, 90), (110, 110), (90, 110))]}
    rotoNode.shapes.value = json.dumps([square])
    rotoNode.links.value = json.dumps([{"input": "tracker", "tracks": ["other"], "frame": 1}])
    rotoNode.outputMode.value = "replaceAlpha"
    graph.update()
    alpha = Evaluator().render(buildPlan(rotoNode), 2).rgba()[:, :, 3]
    # Moved 50 pixels to the right by the Roto link: around x = 150 (rows from the top: y = 100 is row 99)
    assert alpha[99, 150] > 0.9 and alpha[99, 100] < 0.1
