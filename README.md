# forgeneo-mcp

An MCP bridge that lets an AI agent generate images with a local
[Stable Diffusion WebUI Forge Neo](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo)
instance.

The agent writes the prompt and makes the creative decisions. This server only
reports what the instance offers and executes requests faithfully — it never
injects a LoRA, rewrites a prompt, or writes to your models folder.

## What it does

- **Tells the agent what it is working with.** Which checkpoint is loaded, which
  architecture preset is active, whether the model behaves as a turbo/distilled
  variant, the prompt dialect it expects, and whether its VAE and text encoder
  files actually exist.
- **Keeps the architecture preset authoritative.** Sampler, scheduler and shift
  come from the active Forge preset — those describe the model family and are
  not a matter of taste. History adjusts only what an accelerator actually
  changes: step count and guidance. Forge's `anima` preset declares 32 steps at
  CFG 4.0; if your outputs consistently run 10 steps at CFG 1.5 because a turbo
  LoRA is loaded, that is what the agent is told, and it is told why.
- **Flags whether you work with accelerator LoRAs.** A distilled workflow is a
  standing choice, not a per-image one, so the profile reports how often your
  generations load a turbo/dmd2/LCM LoRA and which one, letting the agent pick
  the right default before writing the first prompt.
- **Makes your LoRAs discoverable** by name, title, tags, trigger words and past
  usage — including which ones are accelerators (turbo/distill/dmd2/lcm) that
  change the sampling regime rather than the image.
- **Returns file paths, not base64**, whenever the output folder is readable.

## Requirements

- Forge Neo running with `--api`
- Python 3.10+

## Install

```bash
pip install -e .
```

## Configure

Every setting is optional except the URL if Forge is not on localhost.

| Variable | Purpose | Default |
|---|---|---|
| `FORGE_URL` | Base URL of the instance | `http://127.0.0.1:7860` |
| `FORGE_AUTH` | `user:password` when started with `--api-auth` | none |
| `FORGE_PATH_MAP` | `REMOTE=LOCAL` prefix pairs, `;`-separated, so paths reported by Forge can be read from here | none |
| `FORGE_OUTPUT_DIR` | Output folder, if it cannot be derived automatically | auto |
| `FORGE_TIMEOUT` | Request timeout in seconds | `600` |
| `FORGE_HISTORY_LIMIT` | How many recent outputs to index | `600` |

`FORGE_PATH_MAP` matters when Forge runs on another machine. Forge reports
`I:\sd-webui-forge-neo\...`, which means nothing on the client, so map it to
whatever reaches the same files:

```
FORGE_PATH_MAP=I:\sd-webui-forge-neo=\\desktop-casa\I\sd-webui-forge-neo
```

Without a mapping the bridge still works — it just falls back to decoding the
base64 the API returns.

### Claude Code

```bash
claude mcp add forgeneo \
  -e FORGE_URL=http://127.0.0.1:7860 \
  -- forgeneo-mcp
```

### Any client using `mcp.json`

```json
{
  "mcpServers": {
    "forgeneo": {
      "command": "forgeneo-mcp",
      "env": { "FORGE_URL": "http://127.0.0.1:7860" }
    }
  }
}
```

## Tools

| Tool | Purpose |
|---|---|
| `capabilities` | What this instance offers: routes, counts, which metadata sources are present |
| `model_profile` | The loaded checkpoint, its sampling regime, prompt dialect, and module health |
| `loras` | Search LoRAs by name, title, tag, trigger or description |
| `lora_info` | Full detail for one LoRA, with a ready prompt fragment |
| `models` | List, load or refresh checkpoints |
| `generate` | Generate from an already-written prompt |
| `progress` | Check, interrupt or skip the running job |

## Metadata sources

The LoRA index and model profile are assembled from whatever exists, and each
result says where its facts came from. Nothing is required:

1. `/sdapi/v1/loras` — always available; carries the parsed safetensors header
2. `<name>.json` sidecar — read natively by Forge, whoever wrote it
3. Generation history — your own outputs, with the weights you actually used

On a 362-LoRA collection the header alone supplied a base model for 87%, a title
for 71% and training tags for 48%; sidecars raised tags and descriptions to 99%.
A clean install keeps tier 1 and simply reports less.

## Notes

- `models(action="load")` swaps the checkpoint for the whole instance, including
  anyone using the web UI at that moment.
- Video (Wan) is not wired up: generation runs through `batch_size` in multiples
  of `4n+1` and the API discards the resulting `video_path`. The code path is
  mapped but untested — no video model was available.
- Read-only by design. The server never writes to `models/` or `config.json`.

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check .
```

## License

MIT
