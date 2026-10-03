"""Guarded access to the Grounding-DINO server (port 12181).

Safety rule of the whole mechanism: a failure here means **no information**, never
negative evidence. The GD server client retries and can `SystemExit()` when the
server is down, which would kill an eval run, so every call is wrapped.
"""

import os
import socket
from typing import Any, List, Optional, Tuple

import numpy as np


class GdDetector:
    """Thin, failure-tolerant wrapper around `GroundingDINOClient`."""

    def __init__(self, port: int = 12181, enabled: bool = True) -> None:
        self.port = int(port)
        self.enabled = bool(enabled)
        self._client: Any = None          # None = not created, False = unusable

    def reset(self) -> None:
        """Drop the cached client (process-local; the server keeps running)."""
        self._client = None

    # ---- availability ----
    def available(self) -> bool:
        if not self.enabled:
            return False
        try:
            with socket.create_connection(("127.0.0.1", self.port), timeout=0.5):
                return True
        except OSError:
            return False

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from vlfm.vlm.grounding_dino import GroundingDINOClient

                self._client = GroundingDINOClient(
                    port=int(os.environ.get("GROUNDING_DINO_PORT", self.port))
                )
            except Exception:  # noqa: BLE001
                self._client = False
        return self._client

    # ---- inference ----
    def detect(self, rgb: Optional[np.ndarray], caption: str) -> Optional[Tuple[np.ndarray, np.ndarray, List[str]]]:
        """Return `(boxes_xyxy_normalized, logits, phrases)` or None ("not checked").

        Boxes are normalized xyxy (the VLVM `ObjectDetections` 2D convention), so
        the caller scales them by `[width, height, width, height]` to get pixels.
        """
        if not self.enabled or rgb is None:
            return None
        if not self.available():
            return None
        client = self._get_client()
        if client is False:
            return None
        try:
            det = client.predict(rgb, caption=caption)
        except BaseException:  # noqa: BLE001 - includes SystemExit from the client
            return None
        boxes = det.boxes.detach().cpu().numpy() if hasattr(det.boxes, "detach") else np.asarray(det.boxes)
        logits = det.logits.detach().cpu().numpy() if hasattr(det.logits, "detach") else np.asarray(det.logits)
        return (
            np.asarray(boxes, dtype=np.float64).reshape(-1, 4),
            np.asarray(logits, dtype=np.float64).reshape(-1),
            [str(p) for p in det.phrases],
        )
