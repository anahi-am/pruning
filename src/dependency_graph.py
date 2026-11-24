class DependencyGraph:
    def __init__(self):
        self.concat_map = {12: [10, 6], 15: [13, 4], 18: [16, 13], 21: [19, 10]}
        self.sync_groups = self._build_sync_groups()
        self.layer_dependencies = self._build_layer_dependencies()

    def _build_sync_groups(self) -> list[set[int]]:
        groups = []
        for sources in self.concat_map.values():
            groups.append(set(sources))
        return groups

    def _build_layer_dependencies(self) -> dict[int, int]:
        deps = {}
        sequential_backbone = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        for i in range(len(sequential_backbone) - 1):
            deps[sequential_backbone[i + 1]] = sequential_backbone[i]

        deps[13] = 12
        deps[16] = 15
        deps[17] = 16
        deps[19] = 18
        deps[20] = 19
        deps[22] = 21

        return deps

    def get_sync_partners(self, layer_idx: int) -> set[int]:
        for group in self.sync_groups:
            if layer_idx in group:
                return group - {layer_idx}
        return set()

    def get_input_source(self, layer_idx: int) -> int | None:
        return self.layer_dependencies.get(layer_idx)
