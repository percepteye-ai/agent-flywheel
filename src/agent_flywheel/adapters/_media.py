"""One medium -- an image, a clip, a recording, a file -- as the OTel GenAI
part BOTH SDKs emit.

What a model saw is recorded as OpenTelemetry GenAI message parts. A medium
is a ``blob`` when its bytes are inline (``content``, base64) and a ``uri``
when it is named by an address, keys in that order with ``mime_type`` ``None``
when it is not known: ``{"type", "modality", "mime_type", "content" | "uri"}``.
The modality is the MIME type's major type when that is ``image``, ``video``
or ``audio``, else ``document``. The Node SDK's ``mediaPart``
(``src/model-boundary.js``) is this function in JavaScript; the shared
``media_parts.json`` fixture holds both to the same bytes.

Mission Control's decoder reads every dialect -- these parts, and the
provider shapes they replace -- through one normaliser, and stores the bytes
by reference at its ingest, or withholds them where personal data must be
masked; so an adapter emits what it saw, as it saw it, and never decides
what is kept.

Stdlib only: adapters import no framework at module scope.
"""
from __future__ import annotations

import base64
from typing import Any

__all__ = ["data_url_part", "media_part", "modality_of"]

_MODALITIES = ("image", "video", "audio")


def modality_of(mime_type: Any) -> str:
    """The GenAI modality a MIME type names: its major type when that is
    image, video or audio, else ``document``."""
    major = (mime_type.split("/", 1)[0].strip().lower()
             if isinstance(mime_type, str) else "")
    return major if major in _MODALITIES else "document"


def media_part(*, mime_type: Any = None, data: Any = None, uri: Any = None,
               modality: str | None = None) -> dict[str, Any]:
    """The GenAI part of one medium: a ``blob`` when ``data`` (bytes, or
    their base64 text) is given, else a ``uri``. ``modality`` defaults to
    the MIME type's (:func:`modality_of`)."""
    mime = mime_type if isinstance(mime_type, str) and mime_type else None
    kind = modality or modality_of(mime)
    if isinstance(data, (bytes, bytearray, memoryview)):
        data = base64.b64encode(bytes(data)).decode("ascii")
    if isinstance(data, str):
        return {"type": "blob", "modality": kind, "mime_type": mime, "content": data}
    return {"type": "uri", "modality": kind, "mime_type": mime, "uri": uri}


def data_url_part(url: Any, *, modality: str | None = None,
                  mime_type: Any = None) -> dict[str, Any] | None:
    """A medium named by a URL: a base64 ``data:`` URL is its bytes (a
    ``blob``, as the conventions require), any other URL a ``uri``; ``None``
    for no URL."""
    if not isinstance(url, str) or not url:
        return None
    if url.startswith("data:") and ";base64," in url:
        head, _, data = url.partition(";base64,")
        return media_part(mime_type=head[len("data:"):].split(";")[0] or mime_type,
                          data=data, modality=modality)
    return media_part(mime_type=mime_type, uri=url, modality=modality)
