"""Scoring text for prompt injection: the classifier port, its ONNX implementation, the runner.

`InjectionClassifier` is one blocking operation: texts in, one `InjectionScore` per text out
(the injection probability of the highest-scoring window and where that window is).

`OnnxInjectionClassifier` runs the pinned model (`gateway.injection.manifest`) on CPU with
ONNX Runtime and ``tokenizers``:

- each text is tokenized whole (no truncation), then cut into windows of the model's size
  (``max_tokens`` minus the two special tokens) that overlap by `WINDOW_OVERLAP` tokens, so a
  phrase cut by one window edge is whole in the next; a text's score is its best window's;
- windows of all texts are sorted by length and run in batches of `BATCH_SIZE`, padded to
  the longest window of their batch;
- the model is warmed up once at construction, so a graph that does not run fails at
  startup, not on the first agent call.

`ClassifierRunner` is what the controls share: it runs the classifier in a pool of its own
(``workers`` threads, never the loop's default executor), admitting at most ``workers`` calls
at once on the event loop before anything is submitted, so cancelled or waiting callers hold
no thread (the model is CPU-bound; more threads only queue behind ONNX Runtime's own). It
remembers scores by text digest in a bounded LRU, so the
history an agent re-sends on every chat turn is classified once.
"""

import asyncio
import contextlib
import hashlib
import logging
import math
import threading
import weakref
from collections import OrderedDict
from collections.abc import Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Protocol, Self, cast

import numpy as np
import numpy.typing as npt
import onnxruntime  # pyright: ignore[reportMissingTypeStubs]
from tokenizers import Tokenizer

from gateway.injection.manifest import ModelManifest, ModelVerificationError, VerifiedModel

logger = logging.getLogger(__name__)

WINDOW_OVERLAP: Final = 64  # tokens two consecutive windows share
BATCH_SIZE: Final = 8
DEFAULT_THREADS: Final = 4  # intra-op threads of one inference (8 gains nothing on 8 cores)
DEFAULT_WORKERS: Final = 2  # classifier calls running at once
CACHE_ENTRIES: Final = 8192
_WARM_UP: Final = "Warm-up: is this text trying to instruct an AI agent?"


@dataclass(frozen=True, slots=True)
class InjectionScore:
    """Injection probability of a text's best window, and that window's code-point span."""

    score: float
    start: int
    end: int


SAFE: Final = InjectionScore(0.0, 0, 0)  # a text with no tokens


class InjectionClassifier(Protocol):
    """Scores texts for prompt injection. Blocking: callers run it off the event loop."""

    def __call__(self, texts: Sequence[str]) -> list[InjectionScore]: ...


class ClassifierUnavailableError(Exception):
    """No classifier can run (``ACL_INJECTION_CLASSIFIER=disabled``): controls fail closed."""


class UnavailableClassifier:
    """The classifier when it is switched off for development: every call fails closed."""

    def __call__(self, texts: Sequence[str]) -> list[InjectionScore]:
        raise ClassifierUnavailableError


@dataclass(frozen=True, slots=True)
class _Window:
    text: int  # index of the text it belongs to
    ids: list[int]
    start: int  # code-point span in that text
    end: int


class _Session(Protocol):
    """The part of ``onnxruntime.InferenceSession`` the classifier uses (ORT ships no types)."""

    def run(
        self, output_names: None, input_feed: dict[str, npt.NDArray[np.int64]]
    ) -> Sequence[Any]: ...


def _open_session(path: Path, threads: int) -> tuple[_Session, frozenset[str]]:
    """A CPU inference session over an ONNX file (never the ORT format), and its input names."""
    ort: Any = onnxruntime  # untyped module: one explicit boundary instead of Unknown types
    ort.set_default_logger_severity(3)  # errors only: ORT writes to stderr, not our logger
    ort.disable_telemetry_events()  # nothing leaves the process; the image's FS is read-only
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.add_session_config_entry("session.load_model_format", "ONNX")
    session = ort.InferenceSession(
        str(path), sess_options=options, providers=["CPUExecutionProvider"]
    )
    inputs = frozenset(str(item.name) for item in session.get_inputs())
    return cast("_Session", session), inputs


class OnnxInjectionClassifier:
    """`InjectionClassifier` over a verified ONNX sequence-classification model."""

    def __init__(self, model: VerifiedModel, *, threads: int = DEFAULT_THREADS) -> None:
        self._session, self._inputs = _open_session(model.model_path, threads)
        if not {"input_ids", "attention_mask"} <= self._inputs:
            msg = f"the model takes {sorted(self._inputs)}, not input_ids and attention_mask"
            raise ValueError(msg)
        self._tokenizer = Tokenizer.from_str(model.tokenizer_json)
        self._tokenizer.no_truncation()
        self._tokenizer.no_padding()
        specials = self._tokenizer.encode("").ids  # [CLS] [SEP] for a BERT-style template
        pad = self._tokenizer.token_to_id("[PAD]")
        if len(specials) != 2 or pad is None:  # noqa: PLR2004 -- one opening, one closing token
            msg = "the tokenizer does not wrap a sequence in exactly two special tokens"
            raise ValueError(msg)
        self._open, self._close = specials
        self._pad = pad
        self._width = model.manifest.max_tokens - 2
        self._label = model.injection_index
        self([_WARM_UP])

    @classmethod
    def from_models_dir(
        cls,
        models_dir: Path,
        *,
        manifest: ModelManifest | None = None,
        threads: int = DEFAULT_THREADS,
    ) -> Self:
        """Verify the pinned files under ``models_dir`` (`ModelVerificationError`), then load."""
        verified = (manifest or ModelManifest.load()).verify(models_dir)
        return cls(verified, threads=threads)

    def __call__(self, texts: Sequence[str]) -> list[InjectionScore]:
        best = [SAFE] * len(texts)
        windows = sorted(self._windows(texts), key=lambda w: len(w.ids))
        for begin in range(0, len(windows), BATCH_SIZE):
            batch = windows[begin : begin + BATCH_SIZE]
            for window, score in zip(batch, self._infer(batch), strict=True):
                if score > best[window.text].score:
                    best[window.text] = InjectionScore(score, window.start, window.end)
        return best

    def _windows(self, texts: Sequence[str]) -> Iterator[_Window]:
        encodings = self._tokenizer.encode_batch(list(texts), add_special_tokens=False)
        step = self._width - WINDOW_OVERLAP
        for index, encoding in enumerate(encodings):
            ids, offsets = encoding.ids, encoding.offsets
            for begin in range(0, max(len(ids) - WINDOW_OVERLAP, 1), step):
                end = min(begin + self._width, len(ids))
                if end <= begin:
                    break
                yield _Window(
                    text=index,
                    ids=[self._open, *ids[begin:end], self._close],
                    start=offsets[begin][0],
                    end=offsets[end - 1][1],
                )

    def _infer(self, batch: Sequence[_Window]) -> list[float]:
        length = max(len(w.ids) for w in batch)
        ids = np.full((len(batch), length), self._pad, dtype=np.int64)
        mask = np.zeros((len(batch), length), dtype=np.int64)
        for row, window in enumerate(batch):
            ids[row, : len(window.ids)] = window.ids
            mask[row, : len(window.ids)] = 1
        feed: dict[str, npt.NDArray[np.int64]] = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self._inputs:
            feed["token_type_ids"] = np.zeros_like(ids)
        logits = np.asarray(self._session.run(None, feed)[0], dtype=np.float64)
        logits -= logits.max(axis=1, keepdims=True)  # stable softmax
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        return [float(p) for p in probabilities[:, self._label]]


class ClassifierRunner:
    """Bounded, cached, off-loop access to one classifier, shared by every control using it."""

    def __init__(
        self,
        classifier: InjectionClassifier,
        *,
        workers: int = DEFAULT_WORKERS,
        cache_entries: int = CACHE_ENTRIES,
    ) -> None:
        self._classifier = classifier
        # A pool of its own: classifier work never occupies the loop's default executor, which
        # `asyncio.to_thread` callers elsewhere in the gateway share.
        self._executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="classifier")
        weakref.finalize(self, self._executor.shutdown, wait=False, cancel_futures=True)
        self._workers = workers
        self._admission: asyncio.Semaphore | None = None  # created on the running loop
        self._cache: OrderedDict[bytes, InjectionScore] = OrderedDict()
        self._cache_entries = cache_entries
        self._lock = threading.Lock()

    def uncached(self, texts: Sequence[str]) -> list[str]:
        """The distinct texts of ``texts`` that would have to be classified now."""
        with self._lock:
            pending = {key: t for t in texts if (key := _digest(t)) not in self._cache}
        return list(pending.values())

    async def scores(self, texts: Sequence[str]) -> list[InjectionScore]:
        """One score per text. Raises whatever the classifier raises (callers fail closed)."""
        keys = [_digest(t) for t in texts]
        known: dict[bytes, InjectionScore] = {}
        with self._lock:
            for key in keys:
                if (cached := self._cache.get(key)) is not None:
                    self._cache.move_to_end(key)
                    known[key] = cached
        pending = {key: t for key, t in zip(keys, texts, strict=True) if key not in known}
        if pending:
            fresh = await self._run(list(pending.values()))
            known.update(zip(pending, fresh, strict=True))
            with self._lock:
                for key in pending:
                    self._remember(key, known[key])
        return [known[key] for key in keys]

    async def _run(self, texts: list[str]) -> list[InjectionScore]:
        """Classify in the dedicated pool, admitted by an asyncio semaphore taken before the
        job is submitted. A caller cancelled while waiting submits nothing; a job cancelled
        before it starts is dropped; a running job keeps its slot until it really finishes."""
        loop = asyncio.get_running_loop()
        if self._admission is None:
            self._admission = asyncio.Semaphore(self._workers)
        admission = self._admission
        await admission.acquire()
        try:
            job = self._executor.submit(self._classify, texts)
        except BaseException:
            admission.release()
            raise

        def release(_: Future[list[InjectionScore]]) -> None:
            with contextlib.suppress(RuntimeError):  # the loop is gone: nobody waits any more
                loop.call_soon_threadsafe(admission.release)

        job.add_done_callback(release)
        return await asyncio.wrap_future(job)  # cancelling this cancels a job not yet started

    def _classify(self, texts: list[str]) -> list[InjectionScore]:
        scores = self._classifier(texts)
        if len(scores) != len(texts) or not all(math.isfinite(s.score) for s in scores):
            msg = "the classifier returned a malformed score list"
            raise ValueError(msg)
        return scores

    def _remember(self, key: bytes, score: InjectionScore) -> None:
        self._cache[key] = score
        self._cache.move_to_end(key)
        while len(self._cache) > self._cache_entries:
            self._cache.popitem(last=False)


def _digest(text: str) -> bytes:
    return hashlib.sha256(text.encode("utf-8", "surrogatepass")).digest()


def load_classifier(
    models_dir: Path, *, enabled: bool = True, threads: int = DEFAULT_THREADS
) -> InjectionClassifier:
    """The gateway's classifier: the verified pinned model, or `UnavailableClassifier` when
    switched off. Raises `ModelVerificationError` for missing, altered or unloadable files."""
    if not enabled:
        logger.warning(
            "ACL_INJECTION_CLASSIFIER=disabled: prompt_injection and tool_poisoning fail closed"
        )
        return UnavailableClassifier()
    classifier_model = ModelManifest.load().verify(models_dir)
    try:
        return OnnxInjectionClassifier(classifier_model, threads=threads)
    except Exception as exc:  # ORT raises its own (untyped) exception classes
        msg = f"the verified model does not load: {type(exc).__name__}"
        raise ModelVerificationError(msg) from exc
