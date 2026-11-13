"""Tile dispatch over the C2C GPU worker service (ledger L9.215, A26 4.5 / 5.3).

Implements the small, versioned TILE DISPATCH protocol that tiled samplers in other packs accept (MiniMax H3 Tiled
Diffusion reads ``transformer_options["c2c_tile_dispatch"]``):

    api_version == 1
    map(fns, model_function, device) -> [fns[i](model_function_k, device_k) for every i], in order

Each ``fns[i]`` is a pure per-tile closure; it runs as one task on a worker (its own thread and device) with that
worker's model replica (``replicas[worker_id]``) or, without one, the coordinator's ``model_function``. Results come
back in tile order. Nothing here imports another pack; the protocol is the whole contract.
"""
from __future__ import annotations

from .tasks import TaskManifest, TileService, new_job_id

TILE_DISPATCH_API = 1


class ServiceTileDispatch:
    api_version = TILE_DISPATCH_API
    name = "c2c-service"
    shares_state = False

    def __init__(self, service: TileService, replicas: dict | None = None, timeout: float | None = None):
        self.service, self.replicas, self.timeout = service, dict(replicas or {}), timeout
        if "c2c.tile_dispatch" not in service._executors:
            service.register_executor("c2c.tile_dispatch", self._run)

    def _run(self, task, ctx):
        mf = self.replicas.get(ctx.worker_id, task.payload["model_function"])
        return task.payload["fn"](mf, ctx.device)

    def map(self, fns, model_function, device):
        job = new_job_id("tiles")
        self.service.submit([TaskManifest(job_id=job, task_id=f"tile{i}", order=i, task_type="c2c.tile_dispatch",
                                          tile_or_window_id=str(i),
                                          payload={"fn": fn, "model_function": model_function})
                             for i, fn in enumerate(fns)])
        state = self.service.wait(job, self.timeout)
        info = self.service.job(job)
        if state != "DONE":
            self.service.cancel(job)
            self.service.wait(job, 30)
            self.service.close(job)
            raise RuntimeError(f"tile dispatch ended {state}: {info.get('error')}")
        out = [r.output for r in self.service.results(job)]
        self.service.close(job)
        return out
