"""Which pictures a video file presents: the one definition every count of frames, list of frame times and expectation
of what a decoder must deliver is taken from, so all of them agree on one file.

A video stream's packets are not all pictures it shows. A packet with no data drains the decoder, a packet with no
time cannot be placed, and a packet the container marks discard (the preroll an edit list puts before the visible
start, which the decoder still needs as a reference) is decoded but never shown. Code that copies a stream keeps every
packet; code that counts, places or expects pictures reads presented() alone."""
from __future__ import annotations


def presents(packet) -> bool:
    """Whether a demuxed packet is a picture the file shows."""
    return bool(packet.size) and packet.pts is not None and not packet.is_discard


def presented(container, stream):
    """The packets of `stream` that are pictures the file shows, in the file's own decode order."""
    for packet in container.demux(stream):
        if presents(packet):
            yield packet


def presented_pts(container, stream) -> list[int]:
    """The presentation times of the pictures `stream` shows, in display order."""
    return sorted(int(packet.pts) for packet in presented(container, stream))
