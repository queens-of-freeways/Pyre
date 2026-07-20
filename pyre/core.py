def _ensure_imports():
    import os
    import sys
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)
    import multiprocessing
    multiprocessing.set_start_method("fork", force=True)


def run_generation(
    model="HuggingFaceTB/SmolLM-135M",
    workers=None,
    prompt="Hello, my name is",
    max_tokens=10,
    layers=1,
    discover_timeout=3.0,
    expect_workers=None,
    local_worker=True,
    no_local=False,
):
    _ensure_imports()
    import threading
    import time
    from typing import List, Tuple

    from src.orchestrator.generator import _build_gen, _parse_workers
    from src.orchestrator.worker_node import WorkerNode

    worker_addrs: List[Tuple[str, int]] = []

    if workers:
        worker_addrs = _parse_workers(workers)

    if not workers:
        from src.orchestrator.mdns import discover_workers
        discovered = discover_workers(
            timeout=discover_timeout,
            expect=expect_workers,
        )
        if discovered:
            print(f"Discovered {len(discovered)} worker(s) via mDNS")
            worker_addrs.extend(discovered)

    local_worker_obj = None
    if local_worker and not no_local:
        lw = WorkerNode(host="localhost", port=0, use_mdns=False)
        ready = threading.Event()
        t = threading.Thread(target=lw.start, kwargs={"ready_event": ready}, daemon=True)
        t.start()
        ready.wait()
        worker_addrs.insert(0, ("localhost", lw.port))
        local_worker_obj = lw
        print(f"Local worker started on port {lw.port}")

    if not worker_addrs:
        print("ERROR: No workers available. Start workers or use --workers.")
        return 1

    gen = _build_gen(worker_addrs, model=model, num_layers=layers)

    try:
        gen.generate(prompt, max_tokens=max_tokens, stream=True)
        return 0
    finally:
        gen.root.shutdown()


def start_worker(host="0.0.0.0", port=9000, no_mdns=False):
    _ensure_imports()
    from src.orchestrator.worker_node import WorkerNode

    worker = WorkerNode(host=host, port=port, use_mdns=not no_mdns)
    worker.start()
    return 0


def list_workers(timeout=3.0):
    _ensure_imports()
    from src.orchestrator.mdns import discover_workers

    discovered = discover_workers(timeout=timeout)
    if not discovered:
        print("No workers discovered")
        return 0

    print(f"Found {len(discovered)} worker(s):")
    for host, port in discovered:
        print(f"  {host}:{port}")
    return 0
