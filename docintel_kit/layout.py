"""Layout / structural analysis for PDFs and images.

Detects structural regions (paragraphs, headings, tables, lists, figures,
headers, footers) using `layoutparser <https://layout-parser.github.io/>`_,
which wraps pretrained Detectron2 layout-detection models.

LayoutParser calls are encapsulated behind :class:`BaseLayoutBackend` so the
underlying model (currently a PubLayNet Faster R-CNN model) can be swapped
without changing :func:`analyze_layout`'s signature or return type.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Union

from .types import BlockType, BoundingBox, Document, LayoutBlock, LayoutResult

__all__ = ["BaseLayoutBackend", "LayoutParserBackend", "analyze_layout", "register_layout_backend"]

# Maps the PubLayNet label set used by layoutparser's default model onto our
# coarse BlockType enum.
_PUBLAYNET_LABEL_MAP: dict[str, BlockType] = {
    "Text": BlockType.TEXT,
    "Title": BlockType.TITLE,
    "List": BlockType.LIST,
    "Table": BlockType.TABLE,
    "Figure": BlockType.FIGURE,
}


def _make_document_id(input_: Union[str, bytes], source_path: Optional[str]) -> str:
    if source_path:
        return source_path
    data = input_ if isinstance(input_, bytes) else input_.encode("utf-8")
    return hashlib.sha256(data).hexdigest()[:32]


def _load_images(input_: Union[str, bytes]):
    """Return a list of PIL Images, one per page, for a path or raw bytes."""
    from PIL import Image

    if isinstance(input_, str):
        path = Path(input_)
        if path.suffix.lower() == ".pdf":
            from pdf2image import convert_from_path

            return convert_from_path(str(path))
        return [Image.open(path)]

    if input_.startswith(b"%PDF"):
        from pdf2image import convert_from_bytes

        return convert_from_bytes(input_)

    import io

    return [Image.open(io.BytesIO(input_))]


class BaseLayoutBackend(ABC):
    """Interface for a layout-analysis backend.

    Implementations receive one image per page and must return detected
    regions as :class:`LayoutBlock` objects.
    """

    name: str = "base"

    @abstractmethod
    def detect(self, images: list, document: Document) -> LayoutResult:
        """Detect layout blocks across ``images`` (one PIL Image per page)."""
        raise NotImplementedError


class LayoutParserBackend(BaseLayoutBackend):
    """Default layout backend using layoutparser's pretrained PubLayNet model.

    The Detectron2 model weights are downloaded lazily on first use by
    layoutparser and cached locally afterwards.
    """

    name = "layoutparser"

    def __init__(
        self,
        model_path: str = "lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config",
        weights_path: Optional[str] = None,
        score_threshold: float = 0.5,
    ) -> None:
        self.model_path = model_path
        self.weights_path = weights_path
        self.score_threshold = score_threshold
        self._model = None

    def _get_model(self):
        if self._model is None:
            import layoutparser as lp

            try:
                self._model = lp.Detectron2LayoutModel(
                    self.model_path,
                    model_path=self.weights_path,
                    extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", self.score_threshold],
                    label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"},
                )
            except Exception as exc:
                raise RuntimeError(
                    "Failed to load the layoutparser PubLayNet model. As of this "
                    "writing, layoutparser's built-in model zoo (lp://...) points "
                    "to Dropbox links that return HTML 'File Deleted' pages instead "
                    "of model weights/config (see "
                    "https://github.com/Layout-Parser/layout-parser/issues/168 and "
                    "similar reports) — this is an upstream hosting issue, not a bug "
                    "in docintel_kit. Workarounds: (1) obtain a compatible Detectron2 "
                    "config.yml and model_final.pth from a source you trust, then pass "
                    "their local paths via LayoutParserBackend(model_path=<config path>, "
                    "weights_path=<weights path>) and register it with "
                    "register_layout_backend(), or (2) register an alternative "
                    "BaseLayoutBackend entirely (e.g. a Hugging Face model such as "
                    "microsoft/table-transformer or a YOLO-based layout model)."
                ) from exc
        return self._model

    def detect(self, images: list, document: Document) -> LayoutResult:
        import numpy as np

        model = self._get_model()
        blocks: list[LayoutBlock] = []
        block_counter = 0

        for page_index, image in enumerate(images):
            layout = model.detect(np.array(image))
            # layoutparser returns blocks roughly in detection order; sort
            # top-to-bottom, left-to-right to approximate reading order.
            sorted_blocks = sorted(
                layout, key=lambda b: (round(b.block.y_1, -1), b.block.x_1)
            )
            for order, block in enumerate(sorted_blocks):
                block_type = _PUBLAYNET_LABEL_MAP.get(block.type, BlockType.OTHER)
                blocks.append(
                    LayoutBlock(
                        block_id=f"block-{block_counter}",
                        block_type=block_type,
                        bbox=BoundingBox(
                            x0=float(block.block.x_1),
                            y0=float(block.block.y_1),
                            x1=float(block.block.x_2),
                            y1=float(block.block.y_2),
                            page_index=page_index,
                        ),
                        confidence=float(block.score) if block.score is not None else 1.0,
                        reading_order=order,
                    )
                )
                block_counter += 1

        document.page_count = len(images)
        return LayoutResult(document=document, blocks=blocks, backend=self.name)


_BACKENDS: dict[str, BaseLayoutBackend] = {}


def _get_default_backend() -> BaseLayoutBackend:
    if "layoutparser" not in _BACKENDS:
        _BACKENDS["layoutparser"] = LayoutParserBackend()
    return _BACKENDS["layoutparser"]


def register_layout_backend(backend: BaseLayoutBackend) -> None:
    """Register a custom :class:`BaseLayoutBackend` under its ``name``."""
    _BACKENDS[backend.name] = backend


def analyze_layout(input: Union[str, bytes], backend: str = "layoutparser") -> LayoutResult:
    """Detect structural layout blocks in a PDF or image.

    Args:
        input: A filesystem path or raw bytes for a PDF or image. PDFs are
            rasterized page-by-page before detection.
        backend: Name of a registered :class:`BaseLayoutBackend`. Defaults to
            ``"layoutparser"``, which uses a pretrained PubLayNet model.

    Returns:
        A :class:`LayoutResult` containing detected blocks (paragraphs,
        headings, tables, lists, figures) with bounding boxes and an
        approximate reading order.

    Raises:
        KeyError: if ``backend`` is not registered.
    """
    if backend == "layoutparser":
        selected = _get_default_backend()
    elif backend in _BACKENDS:
        selected = _BACKENDS[backend]
    else:
        raise KeyError(
            f"Unknown layout backend '{backend}'. Registered backends: "
            f"{sorted(set(_BACKENDS) | {'layoutparser'})}"
        )

    source_path = input if isinstance(input, str) else None
    document = Document(
        id=_make_document_id(input, source_path),
        source_path=source_path,
    )
    images = _load_images(input)
    return selected.detect(images, document)
