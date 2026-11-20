# C2C Farm (render farm)

C2C Farm sends a whole workflow to another ComfyUI server, a **backend**, and brings the outputs back. Your own
ComfyUI keeps the queue of farm jobs (the spool), an audit log of every job, and a **C2C Farm** sidebar tab (Live,
Queue, History and Admin views). Three nodes work with it:

| Node | Use it to |
|---|---|
| [C2C Farm Submit — Remote Render](#c2c-farm-submit--remote-render-c2c_submit) | Send a workflow (API format) to a backend, wait for it, fetch the outputs |
| [C2C Farm Cluster Status](#c2c-farm-cluster-status-c2c_clusterstatus) | See which backends answer, how busy they are and how much GPU memory is free; check whether a job would fit |
| [C2C Farm Job History (Audit Log)](#c2c-farm-job-history-audit-log-c2c_jobhistory) | List past jobs, trace one job, compare two jobs |

**Status.** The farm was tested on 2026-07-17 with unit tests and live end to end through a loopback backend (this
ComfyUI submitting to itself: dispatch, render, result download, audit row; commits 3f6e4bf / 903580c). A run on a
remote or cloud backend is not recorded in the commit history: NOT VERIFIED. The 2026-11-19 additions (required
nodes / models check, telemetry, capacity forecast, job trace, job compare; commit 4ebaa55) were tested on the CPU with
stand-ins; the live UI is NOT VERIFIED.

## What a backend is

A backend is a full ComfyUI instance (for example a Docker container, a rented GPU server or a second PC on your
network) that answers the standard ComfyUI HTTP API: `/prompt`, `/history`, `/queue`, `/view`, `/object_info`,
`/system_stats`, `/interrupt`. The farm talks only to that API. The only backend type is `comfyui_native`.

A backend must have every custom node and every model file the workflow uses. A cloud backend may bill GPU time for
every job it runs.

## Setting it up

1. **Your farm user.** Set the environment variable `C2C_USER_NAME` before ComfyUI starts, and declare that name in
   `renderfarm/config/users.json` with a `role` (`admin` or `user`), the `projects` you may submit under (`"*"` = any)
   and `max_concurrent_jobs`. Admins can pause or cancel anyone's jobs in the dashboard; users only their own.
2. **Register the backend** in `renderfarm/config/backends.json`:
   - `name`, `type` (`comfyui_native`), `base_url`;
   - `enabled`: set it to `true` once the backend is reachable (every shipped example is disabled);
   - `api_key_env`: the NAME of an environment variable that holds the backend's token (sent as a Bearer token);
     `""` = no token. The token itself never goes into the file;
   - `max_concurrent_jobs`, and `compute_profiles` (the routing profiles this backend accepts);
   - optional `url_loader_map` and `media_rewrite` (`mapped`, `in-place` or `mounted`), for how local media reaches it
     (step 4).
   The shipped file contains disabled examples, including `local-selftest`, which loops back to this ComfyUI to test
   the plumbing.
3. **Build the backend's ComfyUI** with the same custom nodes. `renderfarm/docker/Dockerfile` builds a ComfyUI image
   and clones the packs listed in `renderfarm/docker/custom_nodes_manifest.txt` at pinned commits.
4. **Local media.** If the workflow loads local images, video or audio, the farm uploads each file once to cloud
   storage and rewrites the workflow so the backend fetches it by URL. Set `C2C_STORAGE_PROVIDER` to `s3`, `r2`,
   `minio`, `azure` or `gcs`, plus that provider's variables (named in the header of each file in
   `renderfarm/storage/`), and install its Python package (`boto3`, `azure-storage-blob` or `google-cloud-storage`).
   A workflow without local media needs no storage.
5. **Python packages** (`renderfarm/requirements.txt`): `requests` is required; `websocket-client` is optional and
   gives live progress and preview images from the backend (without it the farm polls the history).
6. Restart ComfyUI after changing environment variables.

## C2C Farm Submit — Remote Render (`C2C_Submit`)

**Menu:** 🐺 C2C/🧰 Core/Render Farm · **Parameters:** [NODE_REFERENCE](../NODE_REFERENCE.md#c2c_submit)
Needs a configured, enabled backend and `C2C_USER_NAME` (see [Setting it up](#setting-it-up)).

### What it is for

It sends one workflow to a backend through the local spool, waits for it (or not), and downloads the outputs into
your ComfyUI `input/c2c_farm_results/<remote id>/` folder (the backend's own prompt id, shown as `remote_id` in a Job
History trace), where the next node can load them. `result_paths` lists the exact files.

### When to pick it

- To render a workflow on a bigger or idle machine.
- To split one generation over several GPUs of THIS machine, use C2C Multi-GPU Split instead
  ([c2c-distributed.md](c2c-distributed.md)).

### Minimal wiring

```
(text: workflow in API format) ──► prompt_json ──► C2C Farm Submit ──► result_paths ──► (load the files)
                                                                 └─► job_id / info / manifest_json
```

Get `prompt_json` from ComfyUI's **Save (API Format)** export. A bare `{"node_id": {class_type, inputs}}` map and
`{"prompt": {...}}` are both accepted.

### What happens on a run

1. Your user, the backend, the compute profile and the project are checked against the two config files.
2. **Reuse.** The same workflow on the same backend, profile, project and user reuses the job that is queued, running
   or finished, instead of uploading and paying for it again. `force_resubmit` on always spools a new job. Failed and
   cancelled jobs are never reused.
3. **Node check** (always): the backend's `/object_info` must list every node class in the workflow, or the run stops
   before anything is sent.
4. **`dependency_check`** (new 2026-11-19): `off` (default), `report` or `block`. It also compares model files and
   combo values, inputs the backend's version of a node does not know, and required inputs the workflow leaves out.
   `report` writes the result to `manifest_json` and submits anyway; `block` stops before anything is uploaded or
   spooled.
5. Local media are uploaded (step 4 of the setup) and the job is spooled with its `priority` (1-10, higher first).
6. With `wait_for_completion` on, the node shows progress and waits up to `timeout_minutes`.

### Key controls

- `backend`, `compute_profile`: from `backends.json`. The profile is sent with the job so a gateway can route it.
- `priority`, `project_name`: spool order and the project tag in the audit log.
- `wait_for_completion`: off returns the job id at once; follow the job in the C2C Farm sidebar tab.
- `force_resubmit`, `dependency_check`: see above.

### Reading the outputs

- `job_id`: the farm job id, for Job History and the dashboard.
- `result_paths`: the local paths of the fetched outputs, one per line (empty when not waiting).
- `info`: JSON with `job_id`, `backend`, `user`, `compute_profile`, `priority`, `uploaded_media`, `status`,
  `results`, and `reused` when an earlier job was reused.
- `manifest_json`: what the workflow needs: `node_classes` and `model_files` (inputs whose value ends in a model
  extension such as `.safetensors`, `.gguf`, `.onnx`), and `checked`. With `dependency_check` on and a reachable
  backend: `ok`, `missing_classes`, `missing_values`, `unknown_inputs`, `missing_required_inputs`, and a `note` when the
  backend could not report its nodes or the job was reused (not checked again).

### Common mistakes

- **"the C2C_USER_NAME environment variable is not set"** or **"user ... is not declared in users.json"**: set the
  variable before starting ComfyUI and add the user.
- **`<no backends configured>`** in the backend list, or **"backend ... exists but is disabled"**: register the backend
  and set `"enabled": true`.
- **"does not offer compute profile ..."**: pick a profile listed in that backend's `compute_profiles`.
- **"may not submit under project ..."**: the project is not in your `projects` list.
- **"prompt_json is not a workflow"**: paste the API-format export, not the normal workflow file.
- **"Backend missing required nodes. Please update Git repo."**: add the missing packs to
  `custom_nodes_manifest.txt` and rebuild the backend.
- **"requires the API token env var ..."**: the variable named in `api_key_env` is not set.
- **"C2C_STORAGE_PROVIDER is not set"**: the workflow loads local files; configure storage (setup step 4).
- **Timeout**: the node fails after `timeout_minutes`, but the job keeps running on the farm. Look it up in the
  dashboard or with Job History.
- **Cancel** in ComfyUI stops the wait and cancels a job this run submitted on the farm. A reused job belongs to an
  earlier submission and keeps running.
- **`local-selftest` with `wait_for_completion` on**: the job waits behind the very prompt that is waiting for it in the
  single local queue. Use `wait_for_completion` off with that backend.

## C2C Farm Cluster Status (`C2C_ClusterStatus`)

**Menu:** 🐺 C2C/🧰 Core/Render Farm · **Parameters:** [NODE_REFERENCE](../NODE_REFERENCE.md#c2c_clusterstatus)

### What it is for

A snapshot of every registered backend (or one): is it reachable, how many jobs are running and waiting, and GPU
memory. It re-runs about every 5 seconds of queue time.

### Key controls

- `backend`: one backend, or `all`.
- `telemetry` (new 2026-11-19): adds derived numbers per backend and cluster totals.
- `forecast_vram_gb` (0 = off, new 2026-11-19) and `forecast_workers`: for a proposed job that needs this much GPU
  memory and this many free worker slots, says per backend whether it would fit and picks the best backend.

### Reading the outputs

- `cluster_json`: `backends`, one entry each with `reachable`, `running`, `pending`, `gpus` (`name`, `vram_total`,
  `vram_free` as the backend reports them), `max_concurrent_jobs`, `compute_profiles`, and `error` or `disabled`.
  - With `telemetry`: per backend `queue_depth`, `free_workers` (`max_concurrent_jobs` minus running), `vram_total_gb`,
    `vram_free_gb`, `largest_gpu_free_gb`, `vram_used_pct`; and a `cluster` block with totals. CPU, RAM and storage
    are listed under `not_reported`: a backend does not send them.
  - With a forecast: per backend `fits` (true, false, or null when the backend does not report memory), `reason`,
    `queue_ahead`, `free_workers` (and `fits_after_queue` when only the free slots are missing); plus `forecast` with
    `best_backend` and the ranked `candidates`.
- `summary`: one line per backend (running, pending, free workers, free VRAM, forecast) and a line "cluster: N of M
  backend(s) reachable".

### Common mistakes

- **A backend shows UNREACHABLE**: check `base_url`, the network, and its token variable; the `error` field has the
  reason.
- **The forecast is a guess from free memory and free slots**, not a test run of the job.

## C2C Farm Job History (Audit Log) (`C2C_JobHistory`)

**Menu:** 🐺 C2C/🧰 Core/Render Farm · **Parameters:** [NODE_REFERENCE](../NODE_REFERENCE.md#c2c_jobhistory)

### What it is for

It reads the farm's audit log (an SQLite file, `renderfarm/config/audit.db`): who ran what, where, and how long. It
re-runs when the log changes.

### Key controls

- `limit`, `user_filter`, `project_filter`: which recent jobs go into the table.
- `job_id` (new 2026-11-19): the execution trace of one job.
- `compare_job_id` (new 2026-11-19): a second job to compare with `job_id`.

### Reading the outputs

- `stats_json`: `total_jobs`, `by_status`, `by_user`, `completed_with_duration`, `avg_duration_seconds` and
  `total_gpu_seconds` (the sum of job durations), over ALL jobs (the filters do not apply).
- `jobs_table`: the most recent jobs as a text table (job id, user, project, backend, priority, status, duration).
- `job_report` (empty unless `job_id` is set):
  - **Trace:** the job's backend, profile, priority, project, user and status; `stages` with times; `duration_seconds`
    (or `elapsed_seconds` while it runs); `error`. When the job ran in this ComfyUI session, also `live` (remote id,
    progress, node types of the workflow) and `outputs`. `not_recorded` lists what the log does not keep (per-node
    timings, remote logs, resource use).
  - **Compare:** both traces, `config_diff`, `status`, `duration_delta_s`, `duration_ratio`, output counts, and
    `prompt_diff` (nodes added, removed, inputs changed) when both jobs ran in this session.

### Common mistakes

- **"no job ... in the audit log"**: copy the id from Submit's `job_id`, the `jobs_table` or the dashboard.
- **"compare_job_id needs job_id"**: set the first job too.
- **`prompt_diff` says "unavailable"**: the workflows of jobs from an earlier session are not kept; only the
  configuration, status and timing are compared.
