import base64
from types import SimpleNamespace

import pytest
import torch

from skyrl.backends.fireworks.multimodal import (
    ImageSpan,
    decode_image_parts,
    render_supervised_example,
    splice_model_input,
)
from skyrl.backends.fireworks.sft import (
    SFTDatumSpec,
    build_tinker_sft_datums,
    training_batch_to_sft_datum_specs,
)
from skyrl.backends.skyrl_train.training_batch import TrainingInputBatch
from skyrl.train.sft_trainer import collate_sft_batch

tinker = pytest.importorskip("tinker")

_PNG = b"\x89PNG\r\n\x1a\npng"
_JPEG = b"\xff\xd8\xffjpeg"


def _data_uri(data: bytes, mime: str) -> str:
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


def _messages(*parts: dict) -> list[dict]:
    return [{"role": "user", "content": list(parts)}]


def _sample(token_ids, placeholders, loss_mask=None):
    return SimpleNamespace(
        token_ids=token_ids,
        loss_mask=loss_mask if loss_mask is not None else [False] * len(token_ids),
        multi_modal_data=(
            SimpleNamespace(mm_placeholders={"image": placeholders}) if placeholders is not None else None
        ),
    )


def _span(offset: int, length: int, data: bytes = _PNG) -> ImageSpan:
    return ImageSpan(offset, length, data, "png", "hash")


@pytest.mark.parametrize(
    "spans",
    [
        (_span(-1, 1),),
        (_span(0, 0),),
        (_span(3, 2),),
        (_span(0, 2), _span(1, 2)),
        (_span(2, 1), _span(0, 1)),
    ],
)
def test_sft_datum_spec_validates_image_spans(spans):
    with pytest.raises(ValueError):
        SFTDatumSpec(tuple(range(4)), (0, 0, 0, 1), (0.0, 0.0, 0.0, 1.0), spans)


@pytest.mark.parametrize(
    ("offset", "length"),
    [(0, 2), (4, 2), (8, 2)],
    ids=["image-first", "image-middle", "image-last"],
)
def test_splice_model_input_image_positions(offset, length):
    model_input = splice_model_input(list(range(10)), [_span(offset, length)])

    if offset == 0:
        expected_types = ["ImageChunk", "EncodedTextChunk"]
    elif offset + length == 10:
        expected_types = ["EncodedTextChunk", "ImageChunk"]
    else:
        expected_types = ["EncodedTextChunk", "ImageChunk", "EncodedTextChunk"]
    assert model_input.length == 10
    assert [type(chunk).__name__ for chunk in model_input.chunks] == expected_types
    image_chunk = next(chunk for chunk in model_input.chunks if type(chunk).__name__ == "ImageChunk")
    assert isinstance(image_chunk, tinker.types.ImageChunk)
    assert image_chunk.expected_tokens == length


def test_splice_model_input_two_images():
    model_input = splice_model_input(list(range(10)), [_span(1, 2), _span(6, 2, _JPEG)])

    assert [type(chunk).__name__ for chunk in model_input.chunks] == [
        "EncodedTextChunk",
        "ImageChunk",
        "EncodedTextChunk",
        "ImageChunk",
        "EncodedTextChunk",
    ]
    image_chunks = [chunk for chunk in model_input.chunks if isinstance(chunk, tinker.types.ImageChunk)]
    assert [chunk.expected_tokens for chunk in image_chunks] == [2, 2]
    assert model_input.length == 10


def test_splice_model_input_no_images_uses_text_chunk():
    model_input = splice_model_input([1, 2, 3], [])

    assert len(model_input.chunks) == 1
    assert isinstance(model_input.chunks[0], tinker.types.EncodedTextChunk)
    assert model_input.chunks[0].tokens == [1, 2, 3]


def test_render_supervised_example_decodes_images_in_document_order(monkeypatch):
    import renderers

    sample = _sample(
        list(range(10)),
        [SimpleNamespace(offset=6, length=2), SimpleNamespace(offset=1, length=2)],
        [False, False, True, False, False, False, False, False, False, True],
    )
    monkeypatch.setattr(renderers, "build_training_sample", lambda renderer, messages: sample)
    messages = _messages(
        {"type": "image_url", "image_url": {"url": _data_uri(_PNG, "image/jpeg")}},
        {"type": "text", "text": "between"},
        {"type": "image", "image": _data_uri(_JPEG, "image/png")},
    )

    token_ids, loss_mask, spans = render_supervised_example(object(), messages)

    assert token_ids == list(range(10))
    assert loss_mask == sample.loss_mask
    assert [span.data for span in spans] == [_PNG, _JPEG]
    assert [span.image_format for span in spans] == ["png", "jpeg"]
    assert [span.offset for span in spans] == [1, 6]


def test_decode_image_parts_rejects_unsupported_format():
    with pytest.raises(ValueError, match="message 0"):
        decode_image_parts(_messages({"type": "image", "image": _data_uri(b"gif", "image/gif")}))


@pytest.mark.parametrize(
    "placeholders",
    [
        [SimpleNamespace(offset=1, length=2)],
        [SimpleNamespace(offset=1, length=3), SimpleNamespace(offset=3, length=2)],
        [SimpleNamespace(offset=9, length=2)],
    ],
    ids=["count-mismatch", "overlap", "past-end"],
)
def test_render_supervised_example_rejects_malformed_alignment(monkeypatch, placeholders):
    import renderers

    sample = _sample(list(range(10)), placeholders)
    monkeypatch.setattr(renderers, "build_training_sample", lambda renderer, messages: sample)
    messages = _messages(
        {"type": "image", "image": _data_uri(_PNG, "image/png")},
        {"type": "image", "image": _data_uri(_JPEG, "image/jpeg")},
    )

    with pytest.raises(ValueError):
        render_supervised_example(object(), messages)


def test_render_supervised_example_rejects_images_without_multimodal_data(monkeypatch):
    import renderers

    monkeypatch.setattr(
        renderers,
        "build_training_sample",
        lambda renderer, messages: _sample(list(range(4)), None),
    )

    with pytest.raises(ValueError, match="2 images"):
        render_supervised_example(
            object(),
            _messages(
                {"type": "image", "image": _data_uri(_PNG, "image/png")},
                {"type": "image", "image": _data_uri(_JPEG, "image/jpeg")},
            ),
        )


def test_sft_datums_preserve_image_spans_and_shifted_lengths():
    batch = TrainingInputBatch(
        {
            "sequences": torch.tensor([[10, 11, 12, 13, 14, 15]]),
            "attention_mask": torch.ones(1, 6, dtype=torch.long),
            "loss_mask": torch.tensor([[0.0, 0.5, 0.0, 0.5, 0.0, 0.0]]),
        }
    )
    batch.metadata = {"image_spans": [(_span(2, 2),)]}

    specs = training_batch_to_sft_datum_specs(batch)
    datums = build_tinker_sft_datums(batch)

    assert specs[0].image_spans == (_span(2, 2),)
    assert len(specs[0].target_tokens) == len(specs[0].weights) == len(specs[0].model_input_token_ids)
    assert datums[0].model_input.length == len(specs[0].target_tokens) == len(specs[0].weights)
    image_chunk = next(chunk for chunk in datums[0].model_input.chunks if isinstance(chunk, tinker.types.ImageChunk))
    assert image_chunk.expected_tokens == 2


def test_sft_shift_rejects_span_touching_final_token():
    batch = TrainingInputBatch(
        {
            "sequences": torch.tensor([[10, 11, 12, 13]]),
            "attention_mask": torch.ones(1, 4, dtype=torch.long),
            "loss_mask": torch.tensor([[0.0, 1.0, 0.0, 0.0]]),
        }
    )
    batch.metadata = {"image_spans": [(_span(2, 2),)]}

    with pytest.raises(ValueError, match="final token"):
        training_batch_to_sft_datum_specs(batch)


def test_image_span_sidecar_is_row_aligned_through_collation_and_slicing():
    tokenizer = SimpleNamespace(pad_token_id=0)
    examples = [
        {
            "input_ids": [1, 2, 3],
            "attention_mask": [1, 1, 1],
            "num_actions": 3,
            "loss_mask": [0, 1, 0],
            "image_spans": (_span(1, 1),),
        },
        {
            "input_ids": [4, 5],
            "attention_mask": [1, 1],
            "num_actions": 2,
            "loss_mask": [1, 0],
            "image_spans": (),
        },
    ]

    batch = collate_sft_batch(examples, tokenizer)
    sliced = batch.slice(1, 2)

    assert batch.metadata["image_spans"] == [examples[0]["image_spans"], ()]
    assert sliced.metadata["image_spans"] == [()]


def test_image_span_sidecar_survives_batch_repeat_and_concatenation():
    batch = TrainingInputBatch({"sequences": torch.tensor([1, 2])})
    batch.metadata = {"image_spans": [(_span(0, 1),), ()]}
    repeated = batch.repeat_interleave(2)
    other = TrainingInputBatch({"sequences": torch.tensor([3])})
    other.metadata = {"image_spans": [(_span(0, 1, _JPEG),)]}

    concatenated = TrainingInputBatch.cat([repeated, other])

    metadata = concatenated.metadata or {}
    assert metadata["image_spans"] == [
        batch.metadata["image_spans"][0],
        batch.metadata["image_spans"][0],
        (),
        (),
        other.metadata["image_spans"][0],
    ]
