class DependencyGraph:
    def __init__(self):
        # concat_map[layer] = [source1, source2] — inputs to each Concat layer
        # 12: Concat(10, 6)  — C2PSA output + C3k2(6) output
        # 15: Concat(13, 4)  — C3k2(13) output + C3k2(4) output
        # 18: Concat(16, 13) — C3k2(16) output + C3k2(13) output
        # 21: Concat(19, 10) — C3k2(19) output + C2PSA(10) output
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

        # Backbone: sequential 0→1→2→3→4→5→6→7→8→9→10
        for i in range(10):
            deps[i + 1] = i

        # 11: Upsample — takes output of 10
        deps[11] = 10

        # 12: Concat — handled by concat_map, skip
        # 13: C3k2 — takes output of 12 (Concat)
        deps[13] = 12

        # 14: Upsample — takes output of 13
        deps[14] = 13

        # 15: Concat — handled by concat_map, skip
        # 16: C3k2 — takes output of 15 (Concat)
        deps[16] = 15

        # 17: Conv — takes output of 16
        deps[17] = 16

        # 18: Concat — handled by concat_map, skip
        # 19: C3k2 — takes output of 18 (Concat)
        deps[19] = 18

        # 20: Conv — takes output of 19
        deps[20] = 19

        # 21: Concat — handled by concat_map, skip
        # 22: C3k2 — takes output of 21 (Concat)
        deps[22] = 21

        # 23: Detect — takes outputs of 16, 19, 22 (3 scale heads)
        # Not included — Detect is never pruned

        return deps

    def get_sync_partners(self, layer_idx: int) -> set[int]:
        for group in self.sync_groups:
            if layer_idx in group:
                return group - {layer_idx}
        return set()

    def get_input_source(self, layer_idx: int) -> int | None:
        return self.layer_dependencies.get(layer_idx)
