"""Read-only middle-tier shadow logger.

The logger is called from the policy after the nav goal has already been
selected.  Its output is never read by target selection, memory lifecycle,
candidate admission or STOP.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from .features import MiddleTierFeatureAssembler
from .ranker import MiddleTierRanker


class MiddleTierShadowLogger:
    def __init__(
        self,
        *,
        enabled: bool = False,
        assembler: Optional[MiddleTierFeatureAssembler] = None,
        ranker: Optional[MiddleTierRanker] = None,
        every_n: int = 1,
    ) -> None:
        self.enabled = bool(enabled)
        self.assembler = assembler or MiddleTierFeatureAssembler()
        self.ranker = ranker
        self.every_n = max(1, int(every_n))
        self.rows: List[Dict[str, Any]] = []
        self.error: Optional[str] = None

    def reset(self) -> None:
        self.assembler.reset()
        self.rows = []
        self.error = None

    def _score_rows(self, rows: List[Dict[str, Any]]) -> None:
        if self.ranker is None:
            for row in rows:
                row.update(
                    {
                        "s_cross": None,
                        "s_repeat": None,
                        "s_geo": None,
                        "s_ctx": None,
                        "p": None,
                        "rank": None,
                        "model_version": None,
                        "model_available": False,
                    }
                )
            return
        for row in rows:
            try:
                row.update(self.ranker.score(row))
            except Exception as exc:  # noqa: BLE001 - shadow must not kill nav
                self.error = f"{type(exc).__name__}: {exc}"
                row.update(
                    {
                        "s_cross": None,
                        "s_repeat": None,
                        "s_geo": None,
                        "s_ctx": None,
                        "p": None,
                        "rank": None,
                        "model_version": self.ranker.version,
                    }
                )
            row["model_available"] = bool(
                self.ranker is not None and self.ranker.available
            )
        scored = [row for row in rows if row.get("p") is not None]
        scored.sort(key=lambda row: float(row["p"]), reverse=True)
        for rank, row in enumerate(scored):
            row["rank"] = int(rank)
        for row in rows:
            row.setdefault("rank", None)

    def observe(
        self,
        *,
        step: int,
        candidates: Sequence[Dict[str, Any]],
        robot_xy: Any,
        gd_frame: Optional[Dict[str, Any]] = None,
        memory_events: Optional[Sequence[Dict[str, Any]]] = None,
        switch_reason: str = "",
    ) -> List[Dict[str, Any]]:
        if not self.enabled or not candidates:
            return []
        if int(step) % self.every_n != 0:
            return []
        rows = self.assembler.observe(
            step=int(step),
            candidates=candidates,
            robot_xy=robot_xy,
            gd_frame=gd_frame,
            memory_events=memory_events,
            switch_reason=switch_reason,
        )
        self._score_rows(rows)
        self.rows.extend(rows)
        return rows
