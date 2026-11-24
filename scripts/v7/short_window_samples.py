"""Read BP's original gzip JSON array without materializing a whole month's Python objects."""
import gzip
import json


def anchors_from(path):
    decoder = json.JSONDecoder()
    with gzip.open(path, "rt", encoding="utf8") as stream:
        # The writer puts only the scalar horizon, window and scale before this array.
        prefix = stream.read(4096)
        marker = '"anchors":'
        if marker not in prefix:
            raise ValueError(f"Missing anchors array in {path}")
        buf = prefix.split(marker, 1)[1].lstrip()
        if not buf.startswith("["):
            raise ValueError("Invalid anchors array")
        buf = buf[1:]
        first = True
        while True:
            buf = buf.lstrip()
            while not buf:
                buf = stream.read(65536).lstrip()
                if not buf:
                    raise ValueError("Truncated anchors array")
            if buf.startswith("]"):
                if (buf[1:] + stream.read()).strip() != "}":
                    raise ValueError("Invalid end of rollout document")
                return
            if not first:
                if not buf.startswith(","):
                    raise ValueError("Missing anchor separator")
                buf = buf[1:]
            first = False
            while True:
                buf = buf.lstrip()
                try:
                    row, end = decoder.raw_decode(buf)
                    break
                except json.JSONDecodeError:
                    more = stream.read(65536)
                    if not more:
                        raise
                    buf += more
            yield row
            buf = buf[end:]
