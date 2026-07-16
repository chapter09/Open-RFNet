"""DroneRFa labels and the paper's fixed open-set split."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ClassInfo:
    index: str
    label: str
    name: str
    channels: tuple[str, ...]
    outdoor: bool = False


# RF0 is 2.4 GHz and RF1 is 5.8 GHz for the ordinary captures. For T10010
# and T10100, RF0 is 915 MHz and RF1 is 2.4 GHz, as described by DroneRFa.
CLASS_INFO = (
    ClassInfo("A", "T0000", "Background", ("RF0", "RF1")),
    ClassInfo("B", "T0001", "DJI Phantom 3", ("RF0",), True),
    ClassInfo("C", "T0010", "DJI Phantom 4 Pro", ("RF0", "RF1"), True),
    ClassInfo("D", "T0011", "DJI MATRICE 200", ("RF0", "RF1"), True),
    ClassInfo("E", "T0100", "DJI MATRICE 100", ("RF0",), True),
    ClassInfo("F", "T0101", "DJI Air 2S", ("RF0", "RF1"), True),
    ClassInfo("G", "T0110", "DJI Mini 3 Pro", ("RF0", "RF1"), True),
    ClassInfo("H", "T0111", "DJI Inspire 2", ("RF0", "RF1"), True),
    ClassInfo("I", "T1000", "DJI Mavic Pro", ("RF0",), True),
    ClassInfo("J", "T1001", "DJI Mini 2", ("RF0", "RF1"), True),
    ClassInfo("K", "T1010", "DJI Mavic 3", ("RF1",)),
    ClassInfo("L", "T1011", "DJI MATRICE 300", ("RF0", "RF1")),
    ClassInfo("M", "T1100", "DJI Phantom 4 Pro RTK", ("RF1",)),
    ClassInfo("N", "T1101", "DJI MATRICE 30T", ("RF0", "RF1")),
    ClassInfo("O", "T1110", "DJI AVATA", ("RF0", "RF1")),
    ClassInfo("P", "T1111", "DJI DIY", ("RF1",)),
    ClassInfo("Q", "T10000", "DJI MATRICE 600 Pro", ("RF0",)),
    ClassInfo("R", "T10001", "VBar", ("RF0",)),
    ClassInfo("S", "T10010", "FrSky X20", ("RF0", "RF1")),
    ClassInfo("T", "T10011", "Futaba T6IZ", ("RF0",)),
    ClassInfo("U", "T10100", "Taranis Plus", ("RF0",)),
    ClassInfo("V", "T10101", "RadioLink AT9S", ("RF0",)),
    ClassInfo("W", "T10110", "Futaba T14SG", ("RF0",)),
    ClassInfo("X", "T10111", "Skydroid T12", ("RF0",)),
    ClassInfo("Y", "T11000", "Skydroid T10", ("RF0",)),
)

INFO_BY_LABEL = {item.label: item for item in CLASS_INFO}
ALL_LABELS = tuple(item.label for item in CLASS_INFO)

# Table IV in arXiv:2508.12689v2.
KNOWN_LABELS = (
    "T0000", "T0010", "T0011", "T0100", "T0101", "T0110", "T0111",
    "T1000", "T1001", "T1010", "T1011", "T1100", "T1101", "T1110",
    "T1111", "T10000", "T10010", "T10100", "T10101", "T10111",
)
UNKNOWN_LABELS = ("T0001", "T10001", "T10011", "T10110", "T11000")
KNOWN_INDEX = {label: i for i, label in enumerate(KNOWN_LABELS)}


def label_from_filename(filename: str) -> str:
    """Parse the variable-width T label without prefix ambiguity."""
    return filename.split("_", 1)[0]


def paper_channels_for_file(filename: str, label: str) -> tuple[str, ...]:
    """Return only receiver channels carrying the labeled UAV signal.

    DroneRFa records RF0 and RF1 simultaneously. For dual-band devices the
    leading bit of the ``S`` code identifies whether the capture uses the
    initial band (RF0) or the switched band (RF1). Treating the other receiver
    as another labeled example injects background-only samples into the class.
    Single-band devices remain on their sole channel for every ``S`` code.
    Background is intentionally retained from both receivers.
    """
    channels = INFO_BY_LABEL[label].channels
    if label == "T0000" or len(channels) == 1:
        return channels

    try:
        signal_code = filename.rsplit("_S", 1)[1].split(".", 1)[0]
    except IndexError:
        return channels
    if not signal_code or signal_code[0] not in "01":
        return channels
    return ("RF0",) if signal_code[0] == "0" else ("RF1",)


def paper_uses_file(filename: str, label: str) -> bool:
    """The paper uses near-distance outdoor signals and all indoor signals."""
    info = INFO_BY_LABEL[label]
    if info.outdoor or label == "T0000":
        return "_D00_" in filename
    return "_D" not in filename
