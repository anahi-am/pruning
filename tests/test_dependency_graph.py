from src.dependency_graph import DependencyGraph
from tests.helpers import Concat, Detect, TinyConv, TinyModel


def build_model() -> TinyModel:
    return TinyModel(
        [
            TinyConv(4, 8),
            TinyConv(8, 8),
            Concat([-1, 0]),
            TinyConv(16, 8),
            Detect([0, 1, 3]),
        ]
    )


def test_sources_and_concat() -> None:
    graph = DependencyGraph(build_model())

    assert graph.get_sources(0) == ()
    assert graph.get_sources(1) == (0,)
    assert graph.get_sources(2) == (1, 0)
    assert graph.is_concat(2)
    assert not graph.is_concat(1)
    assert graph.get_input_source(1) == 0
    assert graph.get_input_source(2) is None


def test_detect_detection() -> None:
    graph = DependencyGraph(build_model())

    assert graph.detect_idx == 4
    assert graph.detect_sources == (0, 1, 3)


def test_conv_names_and_channels() -> None:
    graph = DependencyGraph(build_model())

    assert graph.get_output_conv_name(0) == "model.0.conv"
    assert graph.get_output_conv_name(2) is None
    assert graph.get_entry_conv_name(1) == "model.1.conv"
    assert graph.get_entry_conv_name(2) is None
    assert graph.get_original_out_channels(0) == 8
    assert graph.get_original_out_channels(2) is None
