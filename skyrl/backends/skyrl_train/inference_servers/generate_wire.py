"""Payload contract for the ``/skyrl/v1/generate`` endpoint.

``VLLMServerActor`` writes these payloads and ``RemoteInferenceClient`` reads
them; nothing else depends on the encoding. Both sides serialize with orjson,
which rejects non-finite floats and has no notion of NumPy arrays, so the
helpers here exist to get sampled logprobs and NumPy side channels across that
boundary intact.

Side-channel arrays use ``{data: <base64>, shape: [...], dtype: <name>}``
envelopes. Keeping ``data`` first lets ``load_packed_body`` decode it from the
raw response without materializing a large Python ``str``.
"""

import math
from collections import deque
from enum import StrEnum
from typing import Any, Collection, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import orjson
import pybase64

from skyrl.backends.skyrl_train.utils.routed_experts import (
    ROUTED_EXPERT_DTYPES,
    RoutedExpertIndices,
    compact_routed_expert_indices,
)
from skyrl.backends.skyrl_train.utils.sample_support import (
    SAMPLE_SUPPORT_DTYPE,
    SAMPLE_SUPPORT_DTYPES,
    SAMPLE_SUPPORT_LOGPROBS_DTYPE,
    CompactSampleSupport,
)

# Matches the floor vLLM applies at its own serving boundaries.
CLAMPED_LOGPROB = -9999.0


class PackedArrayKey(StrEnum):
    """Envelope keys, with ``DATA`` first for ``load_packed_body``."""

    DATA = "data"
    SHAPE = "shape"
    DTYPE = "dtype"


class PackedField(StrEnum):
    """Response-body fields whose value is a packed-array envelope."""

    FULL_LOGPROBS = "full_logprobs"
    ROUTED_EXPERTS = "routed_experts"
    ROLLOUT_SAMPLE_SUPPORT = "rollout_sample_support"
    # float32 sampler logprob of each support member, in the support's member order.
    ROLLOUT_SAMPLE_SUPPORT_LOGPROBS = "rollout_sample_support_logprobs"


class SampleSupportWireKey(StrEnum):
    """Sample-support envelope sidecar fields.

    The envelope's int32 data is the CSR row lengths followed by the member IDs.
    """

    NUM_ROWS = "num_rows"
    TOP_K = "top_k"


PACKED_SIDE_CHANNEL_FIELDS: tuple[str, ...] = tuple(PackedField)

_ENVELOPE_KEYS = frozenset(PackedArrayKey)

_ROUTED_EXPERTS_NDIM = 3
_SAMPLE_SUPPORT_NDIM = 1
_FULL_LOGPROBS_NDIM = 2
FULL_LOGPROBS_DTYPE = np.dtype("<f4")
_FULL_LOGPROBS_DTYPES = frozenset({FULL_LOGPROBS_DTYPE})
_SAMPLE_SUPPORT_LOGPROBS_DTYPES = frozenset({SAMPLE_SUPPORT_LOGPROBS_DTYPE})

_QUOTE = b'"'

# Base64 cannot contain this scan anchor.
_PACKED_DATA_ANCHOR = f':{{"{PackedArrayKey.DATA}":"'.encode()


def build_logprobs_content(
    token_ids: Iterable[int],
    resp_logprobs: Iterable[Optional[Mapping[int, Any]]],
) -> Tuple[list[dict[str, float]], int]:
    """Build ``logprobs.content``, flooring missing and non-finite logprobs.

    vLLM reports a non-finite logprob for a token it just sampled every few
    thousand rollouts, and omits the entry entirely for others. ``isfinite``
    also catches NaN, which vLLM's own ``max(logprob, -9999.0)`` floor misses
    because ``max`` returns its first argument on a False comparison.

    Under ``off_policy_correction.tis_ratio_type="sequence"`` a clamped token
    pins its whole trajectory at the importance-sampling cap; under ``"token"``
    the effect stays bounded to that token.

    Returns the content list and how many entries were clamped.
    """
    content: list[dict[str, float]] = []
    num_clamped = 0
    for tid, lp_dict in zip(token_ids, resp_logprobs):
        # .get over `tid in lp_dict`: an entry present but None would otherwise
        # raise AttributeError instead of taking the floor below.
        entry = lp_dict.get(tid) if lp_dict else None
        logprob = entry.logprob if entry is not None else None
        if logprob is None or not math.isfinite(logprob):
            num_clamped += 1
            logprob = CLAMPED_LOGPROB
        content.append({"logprob": logprob})
    return content, num_clamped


def _to_host_array(routed_experts: Any) -> Any:
    """Bring a framework tensor into host memory, leaving anything else alone.

    vLLM hands back a torch tensor that may still live on a CUDA device, where
    ``np.asarray`` raises instead of transferring. Duck-typed so this module
    stays framework-agnostic, and deliberately not a blanket ``np.asarray``:
    objects without these methods (notably the nested lists this endpoint used
    to send) fall through to ``compact_routed_expert_indices`` and are rejected.
    """
    for method in ("detach", "cpu", "numpy"):
        op = getattr(routed_experts, method, None)
        if callable(op):
            routed_experts = op()
    return routed_experts


def pack_ndarray(
    arr: np.ndarray,
    *,
    allowed_dtypes: Collection[np.dtype],
    extra: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Encode ``arr`` as a base64 envelope carrying ``extra`` as sidecar fields."""
    if not isinstance(arr, np.ndarray):
        raise TypeError("packed array must be a NumPy array")
    if arr.dtype not in allowed_dtypes:
        allowed = sorted(dtype.name for dtype in allowed_dtypes)
        raise ValueError(f"packed array {PackedArrayKey.DTYPE} {arr.dtype.name!r} is not one of {allowed}")
    if extra is not None:
        collisions = sorted(set(extra) & _ENVELOPE_KEYS)
        if collisions:
            raise ValueError(f"sidecar fields collide with envelope keys: {collisions}")

    contiguous = np.ascontiguousarray(arr)
    # `.value` keys: orjson rejects str subclasses as dict keys.
    payload = {
        PackedArrayKey.DATA.value: pybase64.b64encode(memoryview(contiguous)).decode("ascii"),
        PackedArrayKey.SHAPE.value: list(contiguous.shape),
        PackedArrayKey.DTYPE.value: contiguous.dtype.name,
    }
    if extra is not None:
        payload.update(extra)
    return payload


def unpack_ndarray(
    payload: Mapping[str, Any],
    *,
    allowed_dtypes: Collection[np.dtype],
    ndim: int,
) -> Tuple[np.ndarray, dict[str, Any]]:
    """Decode an envelope whose base64 ``data`` may be a string or buffer."""
    if not isinstance(payload, Mapping):
        raise TypeError("packed array payload must be an object")
    try:
        dtype_name = payload[PackedArrayKey.DTYPE]
        shape = tuple(payload[PackedArrayKey.SHAPE])
        data = pybase64.b64decode_as_bytearray(payload[PackedArrayKey.DATA], validate=True)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid packed array envelope: {exc}") from exc

    dtypes = {dtype.name: dtype for dtype in allowed_dtypes}
    if not isinstance(dtype_name, str) or dtype_name not in dtypes:
        raise ValueError(f"packed array {PackedArrayKey.DTYPE} {dtype_name!r} is not one of {sorted(dtypes)}")
    dtype = dtypes[dtype_name]
    # Reject bool, an int subclass; accept np.integer for in-process callers.
    if len(shape) != ndim or any(
        not isinstance(dim, (int, np.integer)) or isinstance(dim, bool) or dim < 0 for dim in shape
    ):
        raise ValueError(f"packed array {PackedArrayKey.SHAPE} {shape} is not {ndim} non-negative dimensions")
    expected_size = math.prod(shape) * dtype.itemsize
    if len(data) != expected_size:
        raise ValueError(
            f"packed array {PackedArrayKey.DATA} has {len(data)} bytes, "
            f"expected {expected_size} for {dtype_name}{list(shape)}"
        )

    array = np.frombuffer(data, dtype=dtype).reshape(shape)
    sidecar = {key: value for key, value in payload.items() if key not in _ENVELOPE_KEYS}
    return array, sidecar


def pack_routed_experts(routed_experts: RoutedExpertIndices) -> dict[str, Any]:
    compact = compact_routed_expert_indices(_to_host_array(routed_experts))
    return pack_ndarray(compact, allowed_dtypes=ROUTED_EXPERT_DTYPES)


def decode_packed_routed_experts(payload: dict[str, Any]) -> RoutedExpertIndices:
    decoded, _ = unpack_ndarray(payload, allowed_dtypes=ROUTED_EXPERT_DTYPES, ndim=_ROUTED_EXPERTS_NDIM)
    compact = compact_routed_expert_indices(decoded)
    if compact.dtype != decoded.dtype:
        raise ValueError(
            f"packed routed_experts uses non-canonical dtype {decoded.dtype.name}; expected {compact.dtype.name}"
        )
    return compact


def pack_sample_support(sample_support: CompactSampleSupport) -> dict[str, Any]:
    lengths = sample_support.row_lengths.astype(SAMPLE_SUPPORT_DTYPE)
    return pack_ndarray(
        np.concatenate([lengths, sample_support.token_ids]),
        allowed_dtypes=SAMPLE_SUPPORT_DTYPES,
        # `.value` keys: orjson rejects str subclasses as dict keys.
        extra={
            SampleSupportWireKey.NUM_ROWS.value: sample_support.num_rows,
            SampleSupportWireKey.TOP_K.value: sample_support.top_k,
        },
    )


def pack_sample_support_logprobs(sample_support: CompactSampleSupport) -> dict[str, Any]:
    if sample_support.logprobs is None:
        raise ValueError("sample support carries no member logprobs to pack")
    return pack_ndarray(sample_support.logprobs, allowed_dtypes=_SAMPLE_SUPPORT_LOGPROBS_DTYPES)


def decode_packed_sample_support(
    payload: dict[str, Any],
    logprobs_payload: Optional[dict[str, Any]] = None,
) -> CompactSampleSupport:
    """Decode support rows, with each member's sampler logprob when ``logprobs_payload`` is given."""
    member_logprobs = None
    if logprobs_payload is not None:
        member_logprobs, _ = unpack_ndarray(
            logprobs_payload, allowed_dtypes=_SAMPLE_SUPPORT_LOGPROBS_DTYPES, ndim=_SAMPLE_SUPPORT_NDIM
        )
    decoded, sidecar = unpack_ndarray(payload, allowed_dtypes=SAMPLE_SUPPORT_DTYPES, ndim=_SAMPLE_SUPPORT_NDIM)
    try:
        num_rows = sidecar[SampleSupportWireKey.NUM_ROWS]
        top_k = sidecar[SampleSupportWireKey.TOP_K]
    except KeyError as exc:
        raise ValueError(f"packed rollout_sample_support carries no {exc.args[0]}") from exc
    if not isinstance(num_rows, int) or isinstance(num_rows, bool) or not 0 <= num_rows <= decoded.size:
        raise ValueError(f"packed rollout_sample_support has invalid {SampleSupportWireKey.NUM_ROWS} {num_rows!r}")
    if not isinstance(top_k, int) or isinstance(top_k, bool):
        raise ValueError(f"packed rollout_sample_support has invalid {SampleSupportWireKey.TOP_K} {top_k!r}")
    # CompactSampleSupport validates the offsets against the member count and top_k, and the
    # logprobs against the member count.
    return CompactSampleSupport.from_token_ids(
        decoded[num_rows:], decoded[:num_rows], top_k=top_k, logprobs=member_logprobs
    )


def _full_logprob_row(row: Any, vocab_size: int) -> np.ndarray:
    if isinstance(row, np.ndarray) or hasattr(row, "__array__"):
        values = np.asarray(row, dtype=FULL_LOGPROBS_DTYPE)
        if values.shape != (vocab_size,):
            raise ValueError(f"full logprob row has shape {values.shape}, expected ({vocab_size},)")
        return values
    if isinstance(row, Mapping):
        if len(row) != vocab_size:
            raise ValueError(f"full logprob row names {len(row)} tokens, expected {vocab_size}")
        try:
            return np.fromiter(
                (row[tid].logprob for tid in range(vocab_size)), dtype=FULL_LOGPROBS_DTYPE, count=vocab_size
            )
        except (KeyError, AttributeError, TypeError) as exc:
            raise ValueError("full logprob row must map every token id to an entry with a logprob") from exc
    raise ValueError(f"unsupported full logprob row type {type(row).__name__}")


def full_logprob_rows(logprobs: Any, token_ids: Sequence[int], vocab_size: int) -> np.ndarray:
    """Return the ``[tokens, vocab_size]`` float32 matrix behind a ``logprobs=-1`` response.

    Accepts vLLM's flat form (``token_ids``/``logprobs`` arrays holding one sampled-first,
    rank-ordered row per position) or one row object per position: a float32 array in token-id
    order, anything exposing ``__array__`` as such an array, or a mapping from token id to an entry
    with a ``logprob`` attribute. Every row must name each token exactly once and be finite.
    """
    num_tokens = len(token_ids)
    if vocab_size <= 0:
        raise ValueError("full logprobs need a positive vocabulary size")
    if any(
        isinstance(tid, bool) or not isinstance(tid, (int, np.integer)) or not 0 <= tid < vocab_size
        for tid in token_ids
    ):
        raise ValueError("response token ids must lie inside the vocabulary")
    rows = np.full((num_tokens, vocab_size), np.nan, dtype=FULL_LOGPROBS_DTYPE)
    flat_ids = getattr(logprobs, "token_ids", None)
    flat_values = getattr(logprobs, "logprobs", None)
    if flat_ids is not None and flat_values is not None and not isinstance(logprobs, (list, tuple)):
        ids = np.asarray(flat_ids)
        values = np.asarray(flat_values, dtype=FULL_LOGPROBS_DTYPE)
        if num_tokens == 0:
            if ids.size:
                raise ValueError("flat logprobs carry rows for an empty response")
            return rows
        if ids.shape != values.shape or ids.ndim != 1 or ids.size % num_tokens:
            raise ValueError("flat logprobs do not divide into one row per response token")
        width = ids.size // num_tokens
        ids = ids.reshape(num_tokens, width)
        values = values.reshape(num_tokens, width)
        if width == vocab_size + 1:
            if not np.array_equal(ids[:, 0], np.asarray(token_ids, dtype=ids.dtype)):
                raise ValueError("sampled-first flat rows do not name the response tokens")
            ids, values = ids[:, 1:], values[:, 1:]
        elif width != vocab_size:
            raise ValueError(f"flat logprob rows have {width} entries for a vocabulary of {vocab_size}")
        if ids.dtype.kind not in "iu" or ids.size and (int(ids.min()) < 0 or int(ids.max()) >= vocab_size):
            raise ValueError("flat logprob rows name a token outside the vocabulary")
        rows[np.arange(num_tokens)[:, None], ids] = values
    else:
        if logprobs is None or len(logprobs) != num_tokens:
            raise ValueError("full logprobs must contain one row per response token")
        for index, row in enumerate(logprobs):
            rows[index] = _full_logprob_row(row, vocab_size)
    if not np.isfinite(rows).all():
        raise ValueError("full logprobs must name every vocabulary token exactly once with a finite value")
    return rows


def pack_full_logprobs(rows: np.ndarray) -> dict[str, Any]:
    """Encode ``[tokens, vocab_size]`` float32 rows without changing a bit."""
    if not isinstance(rows, np.ndarray) or rows.ndim != _FULL_LOGPROBS_NDIM or rows.shape[1] == 0:
        raise ValueError("full logprobs must be a [tokens, vocab_size] array with a non-empty vocabulary")
    rows = rows.astype(FULL_LOGPROBS_DTYPE, copy=False)
    if not np.isfinite(rows).all():
        raise ValueError("full logprobs must be finite")
    return pack_ndarray(rows, allowed_dtypes=_FULL_LOGPROBS_DTYPES)


def decode_packed_full_logprobs(
    payload: Mapping[str, Any], token_ids: Sequence[int], response_logprobs: Optional[Sequence[float]]
) -> np.ndarray:
    """Decode rows and check them against the separately transported sampled logprobs."""
    rows, _ = unpack_ndarray(payload, allowed_dtypes=_FULL_LOGPROBS_DTYPES, ndim=_FULL_LOGPROBS_NDIM)
    if rows.shape[1] == 0:
        raise ValueError("packed full_logprobs has an empty vocabulary")
    if rows.shape[0] != len(token_ids):
        raise ValueError(f"packed full_logprobs has {rows.shape[0]} rows for {len(token_ids)} response tokens")
    if response_logprobs is None or len(response_logprobs) != len(token_ids):
        raise ValueError("full_logprobs require one sampled logprob per response token")
    if not np.isfinite(rows).all():
        raise ValueError("packed full_logprobs must be finite")
    if any(isinstance(tid, bool) or not isinstance(tid, int) or not 0 <= tid < rows.shape[1] for tid in token_ids):
        raise ValueError("response token ids must lie inside the full_logprobs vocabulary")
    sampled = rows[np.arange(len(token_ids)), token_ids]
    if sampled.tobytes() != np.asarray(response_logprobs, dtype=FULL_LOGPROBS_DTYPE).tobytes():
        raise ValueError("sampled logprobs differ from their full_logprobs entries")
    return rows


def _data_prefix(field: str) -> bytes:
    """The bytes an orjson-serialized packed ``field`` opens with."""
    return f'"{field}"'.encode() + _PACKED_DATA_ANCHOR


def load_packed_body(raw: bytes, *, fields: tuple[str, ...] = PACKED_SIDE_CHANNEL_FIELDS) -> dict[str, Any]:
    """Parse a response after replacing registered base64 blobs with views.

    Null fields pass through. An envelope layout the scan cannot splice raises
    instead of falling back to materializing the base64 as a Python string.
    """
    prefixes = {field: _data_prefix(field) for field in fields}
    blobs: dict[str, deque[memoryview]] = {field: deque() for field in fields}
    view = memoryview(raw)
    pieces: list[memoryview] = []
    copied = 0
    scan = 0
    while (anchor := raw.find(_PACKED_DATA_ANCHOR, scan)) >= 0:
        field = _match_packed_field(raw, anchor, prefixes)
        if field is None:
            scan = anchor + len(_PACKED_DATA_ANCHOR)
            continue
        start = anchor + len(_PACKED_DATA_ANCHOR)
        end = raw.find(_QUOTE, start)
        if end < 0:
            raise ValueError(f"unterminated base64 {PackedArrayKey.DATA} for {field} in the response body")
        pieces.append(view[copied:start])
        blobs[field].append(view[start:end])
        copied = scan = end

    if pieces:
        pieces.append(view[copied:])
        body = orjson.loads(b"".join(pieces))
    else:
        body = orjson.loads(raw)
    _restore_packed_data(body, blobs)

    unplaced = {field: len(queue) for field, queue in blobs.items() if queue}
    if unplaced:
        raise ValueError(f"spliced packed blobs found no envelope in the response body: {unplaced}")
    return body


def _match_packed_field(raw: bytes, anchor: int, prefixes: Mapping[str, bytes]) -> Optional[str]:
    """Name the registered field whose prefix ends at ``anchor``, if any."""
    for field, prefix in prefixes.items():
        begin = anchor + len(_PACKED_DATA_ANCHOR) - len(prefix)
        if begin >= 0 and raw.startswith(prefix, begin):
            return field
    return None


def _restore_packed_data(node: Any, blobs: Mapping[str, deque[memoryview]]) -> None:
    """Put each blob back on its envelope's ``data`` key, in document order."""
    if isinstance(node, dict):
        for key, value in node.items():
            queue = blobs.get(key)
            if queue is not None and isinstance(value, dict) and PackedArrayKey.DATA in value:
                if not queue:
                    raise ValueError(f"packed {key} survived the scan unspliced; the response-body layout drifted")
                value[PackedArrayKey.DATA.value] = queue.popleft()
            elif isinstance(value, (dict, list)):
                _restore_packed_data(value, blobs)
    elif isinstance(node, list):
        for item in node:
            if isinstance(item, (dict, list)):
                _restore_packed_data(item, blobs)
