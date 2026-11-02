"""Run inside the isolated OpenCode sandbox container.

The container has the GPU, the network and full root inside, the workspace
mounted read-write, and the object-store credentials — so it can install
libraries and run anything without touching the host. ``nawat sandbox up``
must have started it. The experiment is still approved on the host.
"""

from nawat.lab import P

BLOCK = {"kind": "runtime", "summary": "Inside the isolated OpenCode sandbox (Docker + GPU)"}
RUNTIME = "sandbox"
PARAMS = {
    "container": P("nawat-sandbox", "Name of the running sandbox container"),
}
