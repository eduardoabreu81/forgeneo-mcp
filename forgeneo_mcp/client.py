"""HTTP client for the Forge Neo REST API.

Every call returns an explicit outcome instead of raising, because a bridge is
expected to survive a Forge instance that is booting, busy, or missing a route.
The instance probed during development had /sdapi/v1/cmd-flags returning 500
while every other route worked, so "route exists" and "route works" are treated
as different questions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from .config import Config


@dataclass(frozen=True)
class ApiResult:
    ok: bool
    data: Any = None
    error: str | None = None

    @property
    def value(self) -> Any:
        return self.data if self.ok else None


class ForgeClient:
    """Thin, fault-tolerant wrapper around the Forge REST API."""

    def __init__(self, config: Config) -> None:
        self._config = config
        self._client = httpx.Client(
            base_url=config.url,
            timeout=config.timeout,
            auth=config.auth,
            follow_redirects=True,
        )

    @property
    def config(self) -> Config:
        return self._config

    def close(self) -> None:
        self._client.close()

    def get(self, path: str, timeout: float | None = None) -> ApiResult:
        return self._request("GET", path, timeout=timeout)

    def post(self, path: str, payload: dict[str, Any], timeout: float | None = None) -> ApiResult:
        return self._request("POST", path, json=payload, timeout=timeout)

    def _request(self, method: str, path: str, **kwargs: Any) -> ApiResult:
        timeout = kwargs.pop("timeout", None)
        try:
            response = self._client.request(
                method,
                path,
                timeout=timeout if timeout is not None else self._config.timeout,
                **kwargs,
            )
        except httpx.TimeoutException:
            return ApiResult(False, error=f"timeout calling {path}")
        except httpx.HTTPError as exc:
            return ApiResult(False, error=f"cannot reach {self._config.url}{path}: {exc}")

        if response.status_code >= 400:
            return ApiResult(False, error=f"HTTP {response.status_code} on {path}: {response.text[:200]}")

        try:
            return ApiResult(True, data=response.json())
        except ValueError:
            return ApiResult(False, error=f"non-JSON response from {path}")

    # -- convenience wrappers -------------------------------------------------

    def options(self) -> ApiResult:
        return self.get("/sdapi/v1/options")

    def set_options(self, values: dict[str, Any]) -> ApiResult:
        return self.post("/sdapi/v1/options", values)

    def checkpoints(self) -> ApiResult:
        return self.get("/sdapi/v1/sd-models")

    def loras(self) -> ApiResult:
        return self.get("/sdapi/v1/loras")

    def modules(self) -> ApiResult:
        return self.get("/sdapi/v1/sd-modules")

    def samplers(self) -> ApiResult:
        return self.get("/sdapi/v1/samplers")

    def schedulers(self) -> ApiResult:
        return self.get("/sdapi/v1/schedulers")

    def progress(self) -> ApiResult:
        return self.get("/sdapi/v1/progress", timeout=15.0)

    def interrupt(self) -> ApiResult:
        return self.post("/sdapi/v1/interrupt", {}, timeout=15.0)

    def skip(self) -> ApiResult:
        return self.post("/sdapi/v1/skip", {}, timeout=15.0)

    def openapi(self) -> ApiResult:
        return self.get("/openapi.json")

    def txt2img(self, payload: dict[str, Any]) -> ApiResult:
        return self.post("/sdapi/v1/txt2img", payload)

    def img2img(self, payload: dict[str, Any]) -> ApiResult:
        return self.post("/sdapi/v1/img2img", payload)
