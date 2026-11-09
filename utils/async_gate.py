from __future__ import annotations

import asyncio
from weakref import WeakKeyDictionary


class AsyncConcurrencyGate:
    """Event-loop-safe async semaphore usable across sync factory initialization."""

    def __init__(self, limit: int = 1) -> None:
        self.limit = max(1, int(limit))
        self._semaphores: WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Semaphore] = WeakKeyDictionary()

    def _semaphore(self) -> asyncio.Semaphore:
        loop = asyncio.get_running_loop()
        sem = self._semaphores.get(loop)
        if sem is None:
            sem = asyncio.Semaphore(self.limit)
            self._semaphores[loop] = sem
        return sem

    async def __aenter__(self) -> AsyncConcurrencyGate:
        sem = self._semaphore()
        await sem.acquire()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        sem = self._semaphores.get(asyncio.get_running_loop())
        if sem is not None:
            sem.release()
