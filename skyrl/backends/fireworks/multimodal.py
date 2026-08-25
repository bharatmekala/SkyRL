"""Multimodal rendering helpers for Fireworks SFT."""

from __future__ import annotations

import base64
import binascii
import hashlib
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Sequence, cast

_DATA_URI_RE = re.compile(r"^data:[^;,]+;base64,(?P<data>.*)$", re.DOTALL)
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_JPEG_SIGNATURE = b"\xff\xd8\xff"

if TYPE_CHECKING:
    import tinker


@dataclass(frozen=True)
class ImageSpan:
    """An image's placeholder range and source bytes in a rendered sequence."""

    offset: int
    length: int
    data: bytes
    image_format: Literal["png", "jpeg"]
    sha256: str


def _decode_data_uri(uri: str, message_index: int) -> tuple[bytes, Literal["png", "jpeg"]]:
    match = _DATA_URI_RE.fullmatch(uri)
    if match is None:
        raise ValueError(f"Image in message {message_index} must be a base64 data URI")
    try:
        data = base64.b64decode(match.group("data"), validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError(f"Image in message {message_index} has invalid base64 data") from exc

    if data.startswith(_PNG_SIGNATURE):
        image_format: Literal["png", "jpeg"] = "png"
    elif data.startswith(_JPEG_SIGNATURE):
        image_format = "jpeg"
    else:
        raise ValueError(f"Image in message {message_index} is not a supported PNG or JPEG image")
    return data, image_format


def decode_image_parts(messages: list[dict[str, Any]]) -> list[tuple[bytes, str]]:
    """Decode PNG and JPEG image parts in document order."""

    decoded: list[tuple[bytes, str]] = []
    for message_index, message in enumerate(messages):
        content = message.get("content", []) if isinstance(message, dict) else []
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "image_url":
                image_url = part.get("image_url")
                uri = image_url.get("url") if isinstance(image_url, dict) else None
            elif part.get("type") == "image":
                uri = part.get("image")
            else:
                continue
            if not isinstance(uri, str):
                raise ValueError(f"Image in message {message_index} is missing a data URI")
            decoded.append(_decode_data_uri(uri, message_index))
    return decoded


def _placeholder_value(placeholder, name: str) -> int:
    value = placeholder.get(name) if isinstance(placeholder, dict) else getattr(placeholder, name, None)
    if not isinstance(value, int):
        raise ValueError(f"Image placeholder has invalid {name}: {value!r}")
    return value


def render_supervised_example(
    renderer: Any, messages: list[dict[str, Any]]
) -> tuple[list[int], list[bool], list[ImageSpan]]:
    """Render one supervised example and attach source images to placeholders."""

    import renderers

    sample = renderers.build_training_sample(renderer, cast(Any, messages))
    token_ids = list(sample.token_ids)
    loss_mask = list(sample.loss_mask)
    images = decode_image_parts(messages)
    multi_modal_data = sample.multi_modal_data
    if images and multi_modal_data is None:
        raise ValueError(f"Decoded {len(images)} images but renderer returned no multimodal data")
    placeholders = []
    if multi_modal_data is not None:
        placeholders = list(multi_modal_data.mm_placeholders.get("image", []))
    placeholders.sort(key=lambda placeholder: _placeholder_value(placeholder, "offset"))
    if len(placeholders) != len(images):
        raise ValueError(
            f"Image placeholder/image count mismatch: {len(placeholders)} placeholders, "
            f"{len(images)} decoded images"
        )

    spans: list[ImageSpan] = []
    previous_end = 0
    for placeholder, (data, image_format) in zip(placeholders, images, strict=True):
        offset = _placeholder_value(placeholder, "offset")
        length = _placeholder_value(placeholder, "length")
        end = offset + length
        if offset < 0 or length <= 0 or end > len(token_ids):
            raise ValueError(f"Image span offset={offset} length={length} exceeds token count {len(token_ids)}")
        if offset < previous_end:
            raise ValueError(f"Image spans overlap: previous end={previous_end}, current offset={offset}")
        spans.append(
            ImageSpan(
                offset=offset,
                length=length,
                data=data,
                image_format=cast(Literal["png", "jpeg"], image_format),
                sha256=hashlib.sha256(data).hexdigest(),
            )
        )
        previous_end = end
    return token_ids, loss_mask, spans


def splice_model_input(token_ids: Sequence[int], spans: Sequence[ImageSpan]) -> "tinker.ModelInput":
    """Convert rendered token IDs and image spans into a Tinker input."""

    import tinker
    from tinker.types import EncodedTextChunk, ImageChunk

    token_ids = list(token_ids)
    chunks = []
    cursor = 0
    previous_end = 0
    for span in spans:
        end = span.offset + span.length
        if span.offset < 0 or span.length <= 0 or end > len(token_ids):
            raise ValueError(
                f"Image span offset={span.offset} length={span.length} exceeds token count {len(token_ids)}"
            )
        if span.offset < previous_end:
            raise ValueError(f"Image spans overlap: previous end={previous_end}, current offset={span.offset}")
        if span.offset > cursor:
            chunks.append(EncodedTextChunk(tokens=token_ids[cursor : span.offset]))
        chunks.append(
            ImageChunk(
                data=span.data,
                format=span.image_format,
                expected_tokens=span.length,
            )
        )
        cursor = end
        previous_end = end
    if cursor < len(token_ids):
        chunks.append(EncodedTextChunk(tokens=token_ids[cursor:]))
    model_input = tinker.ModelInput(chunks=cast(Any, chunks))
    if model_input.length != len(token_ids):
        raise AssertionError(f"Spliced ModelInput length {model_input.length} != token count {len(token_ids)}")
    return model_input
