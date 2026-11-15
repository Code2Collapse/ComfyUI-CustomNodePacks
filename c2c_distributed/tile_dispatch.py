"""Tile dispatch over the C2C GPU worker service (ledger L9.215, A26 4.5 / 5.3).

Implements the small, versioned TILE DISPATCH protocol that tiled samplers in other packs accept (MiniMax H3 Tiled
Diffusion reads ``transformer_options["c2c_tile_dispatch"]``):

    api_version == 1
    map(fns, model_function, device) -> [fns[i](model_function_k, device_k) for every i], in order

Each ``fns[i]`` is a pure per-tile closure; it runs as one task on a worker (its own thread and device) with that
worker's model replica (``replicas[worker_id]``) or, without one, the coordinator's ``model_function``. Results come
back in tile order. Nothing here imports another pack; the protocol is the whole contract.

L9.251: the coordinator's ``model_function`` lives on the coordinator's ``device``. A worker on ANOTHER device with no
replica of its own is refused (``map`` before anything is submitted, and again per task) - running it there would be a
wrong-device error or a silent single-GPU run.
"""
from __future__ import annotations

from .tasks import TaskManifest, TileService, new_job_id

TILE_DISPATCH_API = 1


class WrongDevice(ValueError):
    """A worker would run the coordinator's model on a device the model is not on."""


def _same_device(a, b) -> bool:
    import torch
    if b is None:
        return True                         # the caller did not say where its model is: nothing to compare
    try:
        da, db = torch.device(a), torch.device(b)
    except (TypeError, RuntimeError, ValueError):
        return str(a) == str(b)
    if da.type != db.type:
        return False
    if da.type != "cuda":
        return True
    return (da.index or 0) == (db.index or 0)


def _wrong_device_message(bad, workers, device) -> str:
    return (f"C2C tile dispatch: worker(s) {', '.join(f'{w} ({workers[w]})' for w in bad)} have no model replica, "
            f"and the model is on {device}. Running it there would fail with a wrong-device error or quietly use one "
            f"GPU. Give each such worker a replica (replicas={{worker_id: model_function}}) or run the workers on "
            f"{device} only.")


class ServiceTileDispatch:
    api_version = TILE_DISPATCH_API
    name = "c2c-service"
    shares_state = False

    def __init__(self, service: TileService, replicas: dict | None = None, timeout: float | None = None):
        self.service, self.replicas, self.timeout = service, dict(replicas or {}), timeout
        if "c2c.tile_dispatch" not in service._executors:
            service.register_executor("c2c.tile_dispatch", self._run)

    def _run(self, task, ctx):
        mf = self.replicas.get(ctx.worker_id)
        if mf is None:
            if not _same_device(ctx.device, task.payload.get("device")):
                raise WrongDevice(_wrong_device_message([ctx.worker_id], {ctx.worker_id: ctx.device},
                                                        task.payload.get("device")))
            mf = task.payload["model_function"]
        return task.payload["fn"](mf, ctx.device)

    def check_devices(self, device) -> None:
        """Refuse when a live worker without a replica sits on another device than the model (L9.251)."""
        workers = {wid: w.device for wid, w in self.service.workers.items() if getattr(w, "alive", True)}
        bad = sorted(wid for wid, dev in workers.items()
                     if wid not in self.replicas and not _same_device(dev, device))
        if bad:
            raise WrongDevice(_wrong_device_message(bad, workers, device))

    def map(self, fns, model_function, device):
        self.check_devices(device)
        job = new_job_id("tiles")
        self.service.submit([TaskManifest(job_id=job, task_id=f"tile{i}", order=i, task_type="c2c.tile_dispatch",
                                          tile_or_window_id=str(i),
                                          payload={"fn": fn, "model_function": model_function,
                                                   "device": device})
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
