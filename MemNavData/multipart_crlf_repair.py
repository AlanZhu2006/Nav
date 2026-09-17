"""Preserve CRLF pairs across Werkzeug multipart parser chunk boundaries.

This process-local transport repair changes neither uploaded bytes nor model
inputs. Werkzeug's partial-boundary search can retain LF but emit the preceding
CR when a closing delimiter straddles its 64 KiB input chunks. The receiver then
sees an extra CR at the end of the file. Keep the CR with its LF instead.
"""
from functools import wraps


def install():
    from werkzeug.sansio.multipart import MultipartDecoder

    original = MultipartDecoder._last_partial_boundary_index
    if getattr(original, "_memnav_crlf_pair_repair", False):
        return

    @wraps(original)
    def preserve_crlf(self, data):
        index = original(self, data)
        if 0 < index < len(data) and data[index - 1:index + 1] == b"\r\n":
            return index - 1
        return index

    preserve_crlf._memnav_crlf_pair_repair = True
    MultipartDecoder._last_partial_boundary_index = preserve_crlf
