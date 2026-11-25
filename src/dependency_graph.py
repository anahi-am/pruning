from torch import nn
from ultralytics.nn.tasks import DetectionModel

CONCAT_LAYER_TYPE = "Concat"
DETECT_LAYER_TYPE = "Detect"
PREVIOUS_LAYER = -1


class DependencyGraph:
    def __init__(self, model: DetectionModel) -> None:
        self.model = model
        self.layers = model.model
        self.layer_sources = self._build_layer_sources()
        self.detect_idx = self._find_layer(DETECT_LAYER_TYPE)
        self.detect_sources = (
            self.layer_sources[self.detect_idx] if self.detect_idx is not None else ()
        )

    def _find_layer(self, layer_type: str) -> int | None:
        for idx, layer in enumerate(self.layers):
            if type(layer).__name__ == layer_type:
                return idx
        return None

    def _normalize_sources(self, idx: int, f: int | list[int]) -> tuple[int, ...]:
        raw = [f] if isinstance(f, int) else list(f)
        sources = []
        for source in raw:
            resolved = idx - 1 if source == PREVIOUS_LAYER else source
            if 0 <= resolved < idx:
                sources.append(resolved)
        return tuple(sources)

    def _build_layer_sources(self) -> dict[int, tuple[int, ...]]:
        return {
            idx: self._normalize_sources(idx, getattr(layer, "f", PREVIOUS_LAYER))
            for idx, layer in enumerate(self.layers)
        }

    def get_sources(self, idx: int) -> tuple[int, ...]:
        return self.layer_sources.get(idx, ())

    def get_input_source(self, idx: int) -> int | None:
        sources = self.get_sources(idx)
        return sources[0] if len(sources) == 1 else None

    def is_concat(self, idx: int) -> bool:
        return type(self.layers[idx]).__name__ == CONCAT_LAYER_TYPE

    def get_output_conv_name(self, idx: int) -> str | None:
        layer = self.layers[idx]
        cv2 = getattr(layer, "cv2", None)
        if isinstance(cv2, nn.Module) and isinstance(
            getattr(cv2, "conv", None), nn.Conv2d
        ):
            return f"model.{idx}.cv2.conv"
        if isinstance(getattr(layer, "conv", None), nn.Conv2d):
            return f"model.{idx}.conv"
        return None

    def get_entry_conv_name(self, idx: int) -> str | None:
        layer = self.layers[idx]
        cv1 = getattr(layer, "cv1", None)
        if isinstance(cv1, nn.Module) and isinstance(
            getattr(cv1, "conv", None), nn.Conv2d
        ):
            return f"model.{idx}.cv1.conv"
        if isinstance(getattr(layer, "conv", None), nn.Conv2d):
            return f"model.{idx}.conv"
        return None

    def get_original_out_channels(self, idx: int) -> int | None:
        conv_name = self.get_output_conv_name(idx)
        if conv_name is None:
            return None
        return self.model.get_submodule(conv_name).out_channels
