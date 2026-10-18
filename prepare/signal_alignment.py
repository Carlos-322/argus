"""How the reader places numeric rows on footage when their recorded instants do not settle it."""

ALIGNED_ROWS = "row per frame"       # equal row and frame counts, assumed one row per frame
ALIGNED_ASSUMED = "assumed start"    # separate clocks, assumed common start
COARSE_CLOCK = "coarse clock"       # shared stamps, assumed placement within each stamp interval

ALIGNED_CAMERA = "assumed camera clock"  # recorded values on frames whose presentation timing is assumed
RECORDER_ARRIVAL = "recorder arrival"
RECORDER_ARRIVAL_LIMITATION = ("Recorder arrival order does not establish physical event synchronization "
                              "or measured sensor-to-camera delay.")

# Consumers must name the actual placement, without inferring that separate clocks shared a start.
PLACEMENT_TEXT = {
    ALIGNED_ROWS: "placed one row per frame as an assumption",
    ALIGNED_ASSUMED: "placed from both starts",
    COARSE_CLOCK: "placed within each stamp interval as an assumption",
    ALIGNED_CAMERA: "placed on the assumed camera clock",
    RECORDER_ARRIVAL: "placed on the recorder arrival clock; physical synchronization is unverified",
}
UNKNOWN_PLACEMENT = "placed using an unspecified alignment assumption"
PLACEMENT_FIELDS = {
    ALIGNED_ROWS: "placed_row_per_frame",
    ALIGNED_ASSUMED: "placed_from_both_starts",
    COARSE_CLOCK: "placed_within_stamp_intervals",
    ALIGNED_CAMERA: "placed_on_assumed_camera_clock",
    RECORDER_ARRIVAL: "placed_on_recorder_arrival_clock",
}


def placement_text(aligned_by: str | None) -> str:
    """Describe a placement conservatively, including unknown future alignment kinds."""
    return PLACEMENT_TEXT.get(aligned_by, UNKNOWN_PLACEMENT) if aligned_by else ""


def recorder_arrival(context: dict) -> bool:
    """A recorder-log presentation has no independently verified physical event clock."""
    clock = (context.get('source') or {}).get('presentation_clock') or {}
    return clock.get('basis') == 'mcap_log_time' and clock.get('physical_synchronization_verified') is not True
