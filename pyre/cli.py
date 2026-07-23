import os
import sys

import click

from pyre.core import run_generation, start_worker, list_workers


# Auto-forward to pixi when called from the pip-installed entry point
# outside the pixi environment. This lets `pyre` work from anywhere
# while still pulling all conda/pypi deps managed by pixi.
if "PIXI_IN_SHELL" not in os.environ and "PIXI_PROJECT_ROOT" not in os.environ:
    os.environ["PIXI_IN_SHELL"] = "1"
    try:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        os.chdir(project_root)
        os.execvp("pixi", ["pixi", "run", "pyre"] + sys.argv[1:])
    except FileNotFoundError:
        # pixi not installed — proceed with system python
        del os.environ["PIXI_IN_SHELL"]


@click.group()
def main():
    """pyre — distributed LLM inference across machines."""


@main.command()
@click.option("--model", default="HuggingFaceTB/SmolLM-135M",
              help="HuggingFace model ID")
@click.option("--workers", default=None,
              help="Comma-separated list of remote worker IP:PORT")
@click.option("--prompt", default="Hello, my name is",
              help="Input prompt")
@click.option("--max-tokens", default=10, type=int,
              help="Number of tokens to generate")
@click.option("--layers", default=0, type=int,
              help="Number of transformer layers (0 = auto, all layers)")
@click.option("--discover-timeout", default=3.0, type=float,
              help="Seconds to wait for mDNS worker discovery")
@click.option("--expect-workers", default=None, type=int,
              help="Return as soon as N workers found via mDNS")
@click.option("--no-local", is_flag=True, default=False,
              help="Do NOT start a local worker")
@click.option("--temperature", default=0.7, type=float,
              help="Sampling temperature (0 = greedy, 0.7 = default)")
@click.option("--chat", is_flag=True, default=False,
              help="Chat mode — generate until EOS, no need for --max-tokens")
@click.option("--reload", is_flag=True, default=False,
              help="Force re-download model from HuggingFace, bypassing cache")
def run(model, workers, prompt, max_tokens, layers,
        discover_timeout, expect_workers, no_local, temperature, chat, reload):
    """Run distributed generation (auto-starts a local worker)."""
    exit(run_generation(
        model=model,
        workers=workers,
        prompt=prompt,
        max_tokens=max_tokens,
        layers=layers,
        temperature=temperature,
        chat=chat,
        reload=reload,
        discover_timeout=discover_timeout,
        expect_workers=expect_workers,
        no_local=no_local,
    ))


@main.command()
@click.option("--host", default="0.0.0.0", help="Host to bind to")
@click.option("--port", default=9000, type=int, help="Port to listen on")
@click.option("--no-mdns", is_flag=True, default=False,
              help="Disable mDNS registration")
def worker(host, port, no_mdns):
    """Start a remote worker node."""
    exit(start_worker(host=host, port=port, no_mdns=no_mdns))


@main.command()
@click.option("--timeout", default=3.0, type=float,
              help="Seconds to wait for mDNS discovery")
def ps(timeout):
    """List running workers via mDNS."""
    exit(list_workers(timeout=timeout))
