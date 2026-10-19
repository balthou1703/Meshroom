"""
Chapters of a graph: named horizontal lines (`Chapter` nodes) splitting the graph editor into
sections, like the paragraphs of a document. A chapter followed by another one can be folded: the
nodes between the two lines are hidden and everything below moves up, so that the folded line sits
just above the next one.

This module holds the folding geometry, independent of Qt.
"""
from typing import Iterable, List, NamedTuple

# Distance between a folded chapter line and the next one (the line with its title and arrow)
FOLDED_GAP = 36


class FoldBand(NamedTuple):
    """ A folded section: hidden between `start` and `end` (y positions), moving what is below up by `shift`. """
    start: float
    end: float
    shift: float


def foldBands(chapters: Iterable) -> List[FoldBand]:
    """
    Return the folded sections, from top to bottom.

    Args:
        chapters: (y, folded) pairs, one per chapter line, in any order.
    """
    ordered = sorted(chapters, key=lambda chapter: chapter[0])
    bands = []
    for (y, folded), (nextY, _) in zip(ordered, ordered[1:]):
        # The last chapter has no next line to fold onto
        if folded and nextY - y > FOLDED_GAP:
            bands.append(FoldBand(y, nextY, nextY - y - FOLDED_GAP))
    return bands


def foldShift(bands: Iterable[FoldBand], y: float) -> float:
    """ How much an item at `y` moves up because of the folded sections above it. """
    return sum(band.shift for band in bands if y >= band.end)


def isFoldedAway(bands: Iterable[FoldBand], y: float) -> bool:
    """ Whether an item at `y` is inside a folded section (hidden). The chapter lines never are. """
    return any(band.start < y < band.end for band in bands)


def unfoldY(bands: Iterable[FoldBand], displayY: float) -> float:
    """
    Inverse of the folding: the position of an item displayed at `displayY` (e.g. a node created or
    dropped there). A position on a folded line or just below it goes onto the next line, which
    stays visible.
    """
    shift = 0.0
    for band in sorted(bands):
        displayStart = band.start - shift
        displayEnd = band.end - shift - band.shift
        if displayY <= displayStart:
            break
        if displayY < displayEnd:
            return float(band.end)
        shift += band.shift
    return displayY + shift
