"""Media a model saw, recorded as the OTel GenAI parts BOTH SDKs emit.

* ``_media.media_part`` returns each vector of the shared
  ``tests/fixtures/media_parts.json`` -- the Node SDK's ``mediaPart`` returns
  the same, key order included (its own suite; the fixture is held equal
  across the two repositories by ``tests/cross_repo/test_fixture_parity.py``).
* The LangChain adapter records every LangChain spelling of an image, a
  recording, a clip or a file as that part -- a ``blob`` for inline bytes, a
  ``uri`` for an address, a ``file`` for a provider's file id -- and a tool
  result that returned media as its parts; the ADK adapter records Gemini's
  inline and file data the same way.

These are pure functions of the recorded content, so they are tested without
the frameworks installed.
"""
from __future__ import annotations

import json
import pathlib
from types import SimpleNamespace

import pytest

from agent_flywheel.adapters import adk, langchain
from agent_flywheel.adapters._media import data_url_part, media_part, modality_of

FIXTURE = pathlib.Path(__file__).resolve().parent / "fixtures" / "media_parts.json"
VECTORS = json.loads(FIXTURE.read_text(encoding="utf-8"))["vectors"]
PNG = "iVBORw0KGgo="


@pytest.mark.parametrize("vector", VECTORS, ids=lambda v: json.dumps(v["input"])[:60])
def test_every_shared_vector_is_the_part_both_sdks_emit(vector):
    """Mutation: change a key's order, the modality rule, or a missing MIME
    type's ``None`` -> red (and the Node SDK's own copy of this test)."""
    part = media_part(**vector["input"])
    assert list(part.items()) == list(vector["part"].items())


def test_the_modality_is_the_mime_types_major_media_type():
    assert [modality_of(m) for m in ("image/png", "VIDEO/mp4", "audio/wav",
                                     "application/pdf", None, "text/plain")] == [
        "image", "video", "audio", "document", "document", "document"]


def test_a_data_url_is_a_blob_and_any_other_url_a_uri():
    assert data_url_part("data:image/gif;base64,R0lG", modality="image") == {
        "type": "blob", "modality": "image", "mime_type": "image/gif", "content": "R0lG"}
    assert data_url_part("https://x/a.png", modality="image") == {
        "type": "uri", "modality": "image", "mime_type": None, "uri": "https://x/a.png"}
    assert data_url_part(None) is None


IMAGE_BLOB = {"type": "blob", "modality": "image", "mime_type": "image/png", "content": PNG}


@pytest.mark.parametrize("block, part", [
    ({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{PNG}"}}, IMAGE_BLOB),
    ({"type": "image_url", "image_url": "https://x/a.png"},
     {"type": "uri", "modality": "image", "mime_type": None, "uri": "https://x/a.png"}),
    ({"type": "input_audio", "input_audio": {"data": "UklG", "format": "mp3"}},
     {"type": "blob", "modality": "audio", "mime_type": "audio/mpeg", "content": "UklG"}),
    # langchain-core 0.3's standard blocks (``source_type``)...
    ({"type": "image", "source_type": "base64", "data": PNG, "mime_type": "image/png"}, IMAGE_BLOB),
    ({"type": "audio", "source_type": "url", "url": "https://x/a.wav", "mime_type": "audio/wav"},
     {"type": "uri", "modality": "audio", "mime_type": "audio/wav", "uri": "https://x/a.wav"}),
    ({"type": "file", "source_type": "id", "id": "file-1", "mime_type": "application/pdf"},
     {"type": "file", "modality": "document", "mime_type": "application/pdf", "file_id": "file-1"}),
    # ... and 1.x's (``base64`` / ``url`` / ``file_id``).
    ({"type": "video", "base64": "AAAA", "mime_type": "video/mp4"},
     {"type": "blob", "modality": "video", "mime_type": "video/mp4", "content": "AAAA"}),
    ({"type": "image", "url": "gs://b/i.png", "mime_type": "image/png"},
     {"type": "uri", "modality": "image", "mime_type": "image/png", "uri": "gs://b/i.png"}),
    ({"type": "file", "base64": "JVBE", "mime_type": "application/pdf"},
     {"type": "blob", "modality": "document", "mime_type": "application/pdf", "content": "JVBE"}),
    # An Anthropic block passed through by ChatAnthropic.
    ({"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": PNG}},
     IMAGE_BLOB),
])
def test_every_langchain_spelling_of_a_medium_is_its_genai_part(block, part):
    """Mutation: drop any spelling's branch -> red."""
    assert langchain._parts([{"type": "text", "text": "see"}, block]) == [
        {"type": "text", "content": "see"}, part]


def test_an_image_with_no_recorded_address_is_kept_not_dropped():
    """It used to vanish from the recorded input: the model saw an image the
    trace did not show. Kept as recorded, the control plane sees media and
    refuses the decision. Mutation: drop it again -> red."""
    block = {"type": "image_url", "image_url": {"url": None}}
    assert langchain._parts([block]) == [block]


def test_a_tool_result_that_returned_media_is_recorded_as_parts():
    """A LangChain tool result holding an image is its GenAI parts, as the
    Node SDK records one; a result without media is sent as it was."""
    shot = SimpleNamespace(type="tool", tool_call_id="c1", content=[
        {"type": "text", "text": "the page:"},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{PNG}"}}])
    text = SimpleNamespace(type="tool", tool_call_id="c2",
                           content=[{"type": "text", "text": "ok"}])
    assert langchain._message(shot)["parts"][0]["response"] == [
        {"type": "text", "content": "the page:"}, IMAGE_BLOB]
    assert langchain._message(text)["parts"][0]["response"] == text.content


def test_adk_records_gemini_inline_and_file_data_as_genai_parts():
    inline = SimpleNamespace(text=None, function_call=None, function_response=None,
                             inline_data=SimpleNamespace(mime_type="audio/wav", data=b"RIFF"),
                             file_data=None)
    remote = SimpleNamespace(text=None, function_call=None, function_response=None,
                             inline_data=None,
                             file_data=SimpleNamespace(mime_type="video/mp4",
                                                       file_uri="gs://b/v.mp4"))
    document = SimpleNamespace(text=None, function_call=None, function_response=None,
                               inline_data=SimpleNamespace(mime_type="application/pdf",
                                                           data=b"%PDF"),
                               file_data=None)
    assert adk._part(inline) == {"type": "blob", "modality": "audio",
                                 "mime_type": "audio/wav", "content": "UklGRg=="}
    assert adk._part(remote) == {"type": "uri", "modality": "video",
                                 "mime_type": "video/mp4", "uri": "gs://b/v.mp4"}
    assert adk._part(document)["modality"] == "document"
