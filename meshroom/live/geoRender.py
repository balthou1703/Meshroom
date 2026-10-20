"""
Drawing of a geometry over an image, seen through a camera (the Resection node): its shaded surface, its feature edges
(silhouettes, creases, open edges), all its edges, or its points. The hidden parts are removed with a depth buffer
(`rasterize`), the lines are antialiased (each sample spread on its 4 nearest pixels).

Coordinates: the image covers `bbox` (Nuke pixels, y up, row 0 at the top); the camera projects onto the format
(width x height).
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np

MODES = ("shaded", "edges", "wireframe", "points", "none")


def screenPositions(geo, camera, width: int, height: int, bbox):
    """ (u (n,), v (n,), depth (n,), in front (n,)): the vertices in the pixels of the buffer over `bbox` (u: columns,
    v: rows from the top, continuous: the center of pixel (r, c) is (c + 0.5, r + 0.5)). """
    xy, depth, inFront = camera.project(geo.vertices, width, height)
    return xy[:, 0] - bbox.x0, bbox.y1 - xy[:, 1], depth, inFront


def rasterize(u: np.ndarray, v: np.ndarray, depth: np.ndarray, triangles: np.ndarray, shape: Tuple[int, int],
              maxElements: int = 4_000_000, device=None) -> Tuple[np.ndarray, np.ndarray]:
    """
    The depth buffer (h, w) of triangles (inf where there is none) and the index of the triangle seen at each pixel (-1).
    A pixel is covered by a triangle containing its center; a triangle smaller than a pixel covers the pixel of its
    center (no holes in a dense mesh). The depth is interpolated in the image (enough to know what hides what).
    With a torch `device` (the GPU), the pixels of the triangles are found and the nearest kept there.
    """
    fragmentsOf = _triangleFragments if device is None else _torchFragments(device)
    h, w = shape
    if not len(triangles) or h == 0 or w == 0:
        return np.full((h, w), np.inf), np.full((h, w), -1, dtype=np.int64)
    tu, tv, tz = u[triangles], v[triangles], depth[triangles]
    # Reductions over the 3 corners written out (much faster than along a short axis)
    umin, umax = _corners(np.minimum, tu), _corners(np.maximum, tu)
    vmin, vmax = _corners(np.minimum, tv), _corners(np.maximum, tv)
    valid = (_corners(np.minimum, tz) > 1e-9) & np.isfinite(umin + umax + vmin + vmax)
    c0, c1 = np.ceil(umin - 0.5), np.floor(umax - 0.5)
    r0, r1 = np.ceil(vmin - 0.5), np.floor(vmax - 0.5)
    valid &= (umax >= 0) & (umin < w) & (vmax >= 0) & (vmin < h)
    c0, r0 = np.maximum(c0, 0), np.maximum(r0, 0)
    c1, r1 = np.minimum(c1, w - 1), np.minimum(r1, h - 1)
    sizes = np.maximum(c1 - c0 + 1, r1 - r0 + 1)
    fragments = []

    # Triangles covering no pixel center: the pixel of their center
    small = np.flatnonzero(valid & ((c1 < c0) | (r1 < r0)))
    if len(small):
        cu, cv = _corners(np.add, tu[small]) / 3.0, _corners(np.add, tv[small]) / 3.0
        inside = (cu >= 0) & (cu < w) & (cv >= 0) & (cv < h)
        small = small[inside]
        fragments.append((np.floor(cv[inside]).astype(np.int64) * w + np.floor(cu[inside]).astype(np.int64),
                          _corners(np.add, tz[small]) / 3.0, small))
    covering = valid & (c1 >= c0) & (r1 >= r0)
    previous = 0
    for size in (1, 2, 4, 8, 16, 32, 64, 128, 256, None):
        group = np.flatnonzero(covering & (sizes > previous) & ((sizes <= size) if size is not None else True))
        previous = size if size is not None else previous
        if not len(group):
            continue
        if size is None:
            # Large triangles: one at a time over their (clipped) box
            for index in group:
                fragments.append(fragmentsOf(tu[index:index + 1], tv[index:index + 1], tz[index:index + 1],
                                                    c0[index:index + 1], r0[index:index + 1], c1[index:index + 1],
                                                    r1[index:index + 1], int(sizes[index]), w, np.array([index])))
            continue
        step = max(1, maxElements // (size * size))
        for start in range(0, len(group), step):
            chunk = group[start:start + step]
            fragments.append(fragmentsOf(tu[chunk], tv[chunk], tz[chunk], c0[chunk], r0[chunk], c1[chunk],
                                         r1[chunk], size, w, chunk))
    fragments = [f for f in fragments if len(f[0])]
    if device is not None:
        return _torchResolve(fragments, h, w, device)
    zbuffer = np.full(h * w, np.inf)
    ids = np.full(h * w, -1, dtype=np.int64)
    if fragments:
        pixels = np.concatenate([f[0] for f in fragments])
        depths = np.concatenate([f[1] for f in fragments])
        triangleIds = np.concatenate([f[2] for f in fragments])
        # The nearest fragment of each pixel
        order = np.lexsort((depths, pixels))
        pixels, depths, triangleIds = pixels[order], depths[order], triangleIds[order]
        first = np.r_[True, pixels[1:] != pixels[:-1]]
        zbuffer[pixels[first]] = depths[first]
        ids[pixels[first]] = triangleIds[first]
    return zbuffer.reshape(h, w), ids.reshape(h, w)


def _torchFragments(device):
    """ _triangleFragments on a torch device (its results stay there). """
    import torch

    def fragments(tu, tv, tz, c0, r0, c1, r1, size, width, indices):
        def tensor(values, dtype=torch.float32):
            return torch.as_tensor(np.ascontiguousarray(values), dtype=dtype, device=device)
        tu, tv, tz = tensor(tu), tensor(tv), tensor(tz)
        c0, r0, c1, r1 = (tensor(values, torch.int64) for values in (c0, r0, c1, r1))
        cells = torch.arange(size * size, device=device)
        rows = r0[:, None] + torch.div(cells, size, rounding_mode="floor")[None, :]
        columns = c0[:, None] + (cells % size)[None, :]
        px, py = columns.to(torch.float32) + 0.5, rows.to(torch.float32) + 0.5
        ax, ay, bx, by, cx, cy = tu[:, 0:1], tv[:, 0:1], tu[:, 1:2], tv[:, 1:2], tu[:, 2:3], tv[:, 2:3]
        area = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
        safe = torch.where(area.abs() < 1e-12, torch.full_like(area, 1e-12), area)
        w0 = ((bx - px) * (cy - py) - (by - py) * (cx - px)) / safe
        w1 = ((cx - px) * (ay - py) - (cy - py) * (ax - px)) / safe
        w2 = 1.0 - w0 - w1
        inside = (w0 >= -1e-6) & (w1 >= -1e-6) & (w2 >= -1e-6) & (rows <= r1[:, None]) & (columns <= c1[:, None]) \
            & (area.abs() >= 1e-12)
        z = w0 * tz[:, 0:1] + w1 * tz[:, 1:2] + w2 * tz[:, 2:3]
        which = tensor(indices, torch.int64)[:, None].expand_as(inside)
        return (rows[inside] * width + columns[inside], z[inside], which[inside])
    return fragments


def _torchResolve(fragments, h: int, w: int, device):
    """ The nearest fragment of each pixel, on the torch device: (depth buffer, triangle ids) as numpy arrays. """
    import torch

    def tensor(values, dtype):
        return values if torch.is_tensor(values) else torch.as_tensor(np.ascontiguousarray(values), dtype=dtype,
                                                                       device=device)
    zbuffer = torch.full((h * w,), float("inf"), dtype=torch.float32, device=device)
    ids = torch.full((h * w,), -1, dtype=torch.int64, device=device)
    if fragments:
        pixels = torch.cat([tensor(f[0], torch.int64) for f in fragments])
        depths = torch.cat([tensor(f[1], torch.float32).to(torch.float32) for f in fragments])
        triangleIds = torch.cat([tensor(f[2], torch.int64) for f in fragments])
        zbuffer.scatter_reduce_(0, pixels, depths, reduce="amin", include_self=True)
        nearest = depths <= zbuffer[pixels]
        ids.scatter_(0, pixels[nearest], triangleIds[nearest])
    return (zbuffer.to("cpu").numpy().astype(np.float64).reshape(h, w), ids.to("cpu").numpy().reshape(h, w))


def _corners(function, values: np.ndarray) -> np.ndarray:
    """ function (np.minimum, np.maximum, np.add) of the 3 columns of an (m, 3) array. """
    return function(function(values[:, 0], values[:, 1]), values[:, 2])


def _triangleFragments(tu, tv, tz, c0, r0, c1, r1, size: int, width: int, indices):
    """ The pixels (flat indices), depths and triangle indices of the pixel centers inside triangles (n of them),
    tested over a size x size grid from their box corner. """
    dr, dc = np.divmod(np.arange(size * size), size)
    rows = r0[:, None] + dr[None, :]
    columns = c0[:, None] + dc[None, :]
    px, py = columns + 0.5, rows + 0.5
    ax, ay, bx, by, cx, cy = tu[:, 0:1], tv[:, 0:1], tu[:, 1:2], tv[:, 1:2], tu[:, 2:3], tv[:, 2:3]
    area = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
    safe = np.where(np.abs(area) < 1e-12, 1e-12, area)
    w0 = ((bx - px) * (cy - py) - (by - py) * (cx - px)) / safe
    w1 = ((cx - px) * (ay - py) - (cy - py) * (ax - px)) / safe
    w2 = 1.0 - w0 - w1
    epsilon = -1e-7
    inside = (w0 >= epsilon) & (w1 >= epsilon) & (w2 >= epsilon) & (rows <= r1[:, None]) & (columns <= c1[:, None]) \
        & (np.abs(area) >= 1e-12)
    z = w0 * tz[:, 0:1] + w1 * tz[:, 1:2] + w2 * tz[:, 2:3]
    which = np.broadcast_to(np.asarray(indices)[:, None], inside.shape)
    return ((rows[inside] * width + columns[inside]).astype(np.int64), z[inside], which[inside])


def triangleNormals(geo) -> np.ndarray:
    """ The unit normals (m, 3) of the triangles of a geometry (0 for degenerate ones). """
    triangles = geo.triangles()
    v0, v1, v2 = (geo.vertices[triangles[:, i]] for i in range(3))
    normals = np.cross(v1 - v0, v2 - v0)
    lengths = np.sqrt(_dot(normals, normals))[:, None]
    return normals / np.where(lengths < 1e-30, 1.0, lengths)


def _dot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """ The dot products of the rows of two (m, 3) arrays. """
    return a[:, 0] * b[:, 0] + a[:, 1] * b[:, 1] + a[:, 2] * b[:, 2]


def featureEdges(geo, cameraPosition: np.ndarray, creaseDegrees: float = 40.0,
                 normals: Optional[np.ndarray] = None) -> np.ndarray:
    """
    The edges (k, 2) that draw the shape of a mesh: silhouettes (between a triangle facing the camera and one facing
    away), creases (the triangles on both sides at more than `creaseDegrees`) and open edges (one triangle).
    """
    edges, sides = geo.topology.triangleEdges()
    if not len(edges):
        return edges
    normals = triangleNormals(geo) if normals is None else normals
    triangles = geo.triangles()
    facing = _dot(cameraPosition - geo.vertices[triangles[:, 0]], normals) > 0
    open_ = sides[:, 1] < 0
    first, second = sides[:, 0], np.where(open_, sides[:, 0], sides[:, 1])
    crease = _dot(normals[first], normals[second]) < math.cos(math.radians(creaseDegrees))
    silhouette = facing[first] != facing[second]
    return edges[open_ | crease | silhouette]


def _segmentSamples(u, v, z, edges: np.ndarray, shape, maxSamples: int):
    """ Samples every pixel along edges (both ends in front of the camera): (u, v, depth). """
    h, w = shape
    a, b = edges[:, 0], edges[:, 1]
    du, dv = u[b] - u[a], v[b] - v[a]
    # Edges far outside the image are skipped
    margin = 2.0
    outside = ((u[a] < -margin) & (u[b] < -margin)) | ((u[a] > w + margin) & (u[b] > w + margin)) | \
              ((v[a] < -margin) & (v[b] < -margin)) | ((v[a] > h + margin) & (v[b] > h + margin))
    keep = ~outside
    a, b, du, dv = a[keep], b[keep], du[keep], dv[keep]
    # Two samples per pixel: solid antialiased lines
    lengths = np.minimum(np.ceil(2.0 * np.maximum(np.abs(du), np.abs(dv))).astype(np.int64) + 1, 8 * max(w, h))
    total = int(lengths.sum())
    if total > maxSamples:
        chosen = np.linspace(0, len(a) - 1, max(1, int(len(a) * maxSamples / total))).astype(np.int64)
        a, b, du, dv, lengths = a[chosen], b[chosen], du[chosen], dv[chosen], lengths[chosen]
    segment = np.repeat(np.arange(len(a)), lengths)
    firsts = np.r_[0, np.cumsum(lengths)[:-1]]
    t = (np.arange(len(segment)) - firsts[segment]) / np.maximum(lengths[segment] - 1, 1)
    return (u[a][segment] + du[segment] * t, v[a][segment] + dv[segment] * t,
            z[a][segment] + (z[b] - z[a])[segment] * t)


def _splat(coverage: np.ndarray, su: np.ndarray, sv: np.ndarray, weight=1.0):
    """ Antialiased points: each sample (continuous pixel position) spread on its 4 nearest pixels (bilinear). """
    h, w = coverage.shape
    x, y = su - 0.5, sv - 0.5
    x0, y0 = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
    fx, fy = x - x0, y - y0
    flat = coverage.reshape(-1)
    for dx, dy, wx, wy in ((0, 0, 1 - fx, 1 - fy), (1, 0, fx, 1 - fy), (0, 1, 1 - fx, fy), (1, 1, fx, fy)):
        columns, rows = x0 + dx, y0 + dy
        inside = (columns >= 0) & (columns < w) & (rows >= 0) & (rows < h)
        np.add.at(flat, rows[inside] * w + columns[inside], (wx * wy * weight)[inside])


def _composite(rgba: np.ndarray, coverage: np.ndarray, color, alpha: float):
    """ `color` (rgb, or an (h, w, 3) array) over the image where `coverage` (h, w) is, with `alpha`. """
    amount = (np.clip(coverage, 0.0, 1.0) * alpha).astype(np.float32)[:, :, None]
    color = np.asarray(color, np.float32)
    rgba[:, :, :3] = rgba[:, :, :3] * (1.0 - amount) + color[..., :3] * amount
    rgba[:, :, 3:4] = np.maximum(rgba[:, :, 3:4], amount)


def drawLines(rgba: np.ndarray, u, v, z, edges: np.ndarray, color, alpha: float, zbuffer: Optional[np.ndarray] = None,
              maxSamples: int = 20_000_000, tolerance: float = 0.004):
    """ Draw edges (vertex index pairs) antialiased; with a depth buffer, the parts behind the surface are hidden. """
    if not len(edges):
        return
    shape = rgba.shape[:2]
    su, sv, sz = _segmentSamples(u, v, z, edges, shape, maxSamples)
    if zbuffer is not None and len(su):
        h, w = shape
        columns = np.clip(np.floor(su).astype(np.int64), 0, w - 1)
        rows = np.clip(np.floor(sv).astype(np.int64), 0, h - 1)
        # On the surface (the same depth within the tolerance) or in front of it
        visible = sz <= zbuffer[rows, columns] * (1.0 + tolerance) + 1e-9
        su, sv = su[visible], sv[visible]
    coverage = np.zeros(shape, np.float32)
    _splat(coverage, su, sv)
    _composite(rgba, coverage, color, alpha)


def drawGeometry(rgba: np.ndarray, bbox, geo, camera, width: int, height: int, mode: str = "shaded",
                 color=(0.35, 1.0, 0.45), alpha: float = 0.6, creaseDegrees: float = 40.0, hiddenLines: bool = True,
                 maxEdges: int = 2_000_000, pointSize: int = 2, useGpu: bool = False) -> np.ndarray:
    """
    Draw a geometry over an RGBA array covering `bbox`, in place (see MODES):
        shaded: its surface lit from the camera, and its feature edges;
        edges: its feature edges (silhouettes, creases at more than `creaseDegrees`, open edges);
        wireframe: all the edges of its faces;
        points: its vertices.
    `hiddenLines`: the edges (and points) behind the surface are not drawn. `useGpu`: the depth buffer is computed on
    the GPU when there is one (meshroom/live/gpu.py).
    """
    from . import gpu

    def depthBuffer(u, v, z, triangles, shape):
        return gpu.run(useGpu, lambda device: rasterize(u, v, z, triangles, shape, device=device),
                       lambda: rasterize(u, v, z, triangles, shape), "Geometry depth buffer")
    if geo is None or geo.isEmpty or mode == "none":
        return rgba
    shape = rgba.shape[:2]
    u, v, z, inFront = screenPositions(geo, camera, width, height, bbox)
    hasFaces = len(geo.faceCounts) > 0
    color = np.asarray(color, np.float32)
    if mode == "points" or not hasFaces:
        if mode == "points" and hasFaces and hiddenLines:
            zbuffer, _ = depthBuffer(u, v, z, geo.triangles(), shape)
            h, w = shape
            columns = np.clip(np.floor(u).astype(np.int64), 0, w - 1)
            rows = np.clip(np.floor(v).astype(np.int64), 0, h - 1)
            inFront = inFront & (z <= zbuffer[rows, columns] * 1.004 + 1e-9)
        coverage = np.zeros(shape, np.float32)
        for dx in range(-(pointSize // 2), pointSize - pointSize // 2):
            for dy in range(-(pointSize // 2), pointSize - pointSize // 2):
                _splat(coverage, u[inFront] + dx, v[inFront] + dy)
        _composite(rgba, coverage, color, alpha)
        return rgba
    needDepth = mode == "shaded" or hiddenLines
    zbuffer, ids = depthBuffer(u, v, z, geo.triangles(), shape) if needDepth else (None, None)
    if mode == "shaded":
        # Lit from the camera, both sides: bright where the surface faces it, darker at grazing angles
        normals = triangleNormals(geo)
        triangles = geo.triangles()
        corners = geo.vertices
        toCamera = camera.position - (corners[triangles[:, 0]] + corners[triangles[:, 1]] + corners[triangles[:, 2]]) / 3.0
        toCamera /= np.maximum(np.sqrt(_dot(toCamera, toCamera)), 1e-30)[:, None]
        light = 0.35 + 0.65 * np.abs(_dot(normals, toCamera))
        seen = ids >= 0
        shade = np.where(seen, light[np.maximum(ids, 0)], 0.0)
        _composite(rgba, seen.astype(np.float32), color[None, None, :3] * shade[:, :, None].astype(np.float32), alpha)
        edges = featureEdges(geo, camera.position, creaseDegrees, normals)
        drawLines(rgba, u, v, z, _inFrontEdges(edges, inFront, maxEdges), np.minimum(color * 1.25, 1.0), min(1.0, alpha * 1.4),
                  zbuffer)
    elif mode == "edges":
        edges = featureEdges(geo, camera.position, creaseDegrees)
        drawLines(rgba, u, v, z, _inFrontEdges(edges, inFront, maxEdges), color, alpha, zbuffer)
    else:
        drawLines(rgba, u, v, z, _inFrontEdges(geo.edges(), inFront, maxEdges), color, alpha, zbuffer)
    return rgba


def _inFrontEdges(edges: np.ndarray, inFront: np.ndarray, maxEdges: int) -> np.ndarray:
    edges = edges[inFront[edges[:, 0]] & inFront[edges[:, 1]]] if len(edges) else edges
    if len(edges) > maxEdges:
        edges = edges[np.linspace(0, len(edges) - 1, maxEdges).astype(np.int64)]
    return edges
