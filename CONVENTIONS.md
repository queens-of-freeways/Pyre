# Mojo/MAX coding conventions
- Mojo files use `.mojo` extension. Use `fn`, `var`, `let`, `struct`, `@value`.
- NEVER use Python syntax in .mojo files. NEVER use Mojo syntax in .py files.
- Graph construction is Python only. Custom ops are Mojo only.
- All network code is Mojo (kernels/comm/). Use non-blocking sends before recv.
- Every public fn has a mojo test in tests/.
- Commit message format: "phase{N}: {what}" — never commit broken builds.
- If a max.graph API call fails to compile, read docs/llms-python.txt before guessing.
