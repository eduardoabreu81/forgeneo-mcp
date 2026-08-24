# 🌉 Forge Neo MCP

<div align="center">

[![Forge Neo](https://img.shields.io/badge/Forge-Neo-blue)](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo)
[![MCP](https://img.shields.io/badge/MCP-stdio-orange)](https://modelcontextprotocol.io)
[![Python](https://img.shields.io/badge/Python-3.10%2B-yellow)](https://www.python.org/)
[![Version](https://img.shields.io/badge/Version-0.1.0-brightgreen)](https://github.com/eduardoabreu81/forgeneo-mcp)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **MCP server for [Stable Diffusion WebUI Forge - Neo](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo)** · Give any MCP-capable AI agent the ability to generate images on your own GPU

</div>

Lets Claude, or any agent that speaks MCP, drive your local Forge Neo — reading which checkpoint is loaded, the sampling parameters that actually work for it, the prompt dialect it expects, and your LoRA library, then generating from a prompt the agent wrote itself.

> [!Important]
> Forge Neo must be started with `--api`. No extension, no custom node, no changes to your Forge installation — the bridge is read-only and talks to the REST API that Forge already exposes.

---

## 📋 Table of Contents

- [Why it exists](#-why-it-exists)
- [Features](#-features)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Tools](#-tools)
- [How it decides](#-how-it-decides)
- [Roadmap](#-roadmap)
- [Credits](#-credits)

---

## 🎯 Why it exists

Wrapping `/sdapi/v1/txt2img` in a tool takes an afternoon. The hard part is everything the agent needs to know *before* writing the prompt.

A turbo Anima checkpoint wants 10 steps at CFG 1.5 with booru tags. A Krea 2 wants 8 steps at CFG 1.0 and full sentences — quality tags actively hurt it. An Illustrious needs `masterpiece, best quality` or it visibly degrades. None of this is in the API, and getting it wrong never fails loudly; it just produces worse images.

This bridge works it out from the running instance, and says how it knows.

> Every number the agent receives is measured in **your** environment — your history, your instance's options, your checkpoint's tags. Nothing is a statistic borrowed from someone else's install.

---

## ✨ Features

> ⭐ = beyond what a plain API wrapper gives you

### 🖼️ Generation
- **txt2img and img2img** from a prompt the agent wrote, sent verbatim
- **Never injects a LoRA on its own** — discovery is a separate tool, the decision stays with the agent ⭐
- Returns **file paths** rather than base64, so a batch does not flood the agent's context ⭐
- Sends every parameter that matters explicitly, instead of letting API defaults leak in ⭐

### 🧭 Model profile ⭐
- Which checkpoint and architecture preset are live, and the checkpoint's own tags
- **Sampling regime measured from your own outputs**, keyed by checkpoint *and* accelerator LoRA
- **Turbo detection as a tri-state** with evidence — `unknown` is a real answer, not a silent `false`
- Validates that the preset's VAE and text encoder files actually exist
- Flags whether you habitually work with accelerator LoRAs

### 🗣️ Prompt dialects ⭐
- Per-lineage **quality tags**: Pony's score ladder, Illustrious booru tags, Animagine's ordered template, Anima's hybrid vocabulary — and prose models that must get **no tags at all**
- Tag spelling, artist syntax, emphasis weighting, and what each model is *not* for
- Resolved from cache, an optional CivitAI lookup, your own past prompts, the architecture, or your LoRA library — reporting which

### 🎨 LoRA library ⭐
- Search by name, title, tag, trigger word or description — **on demand**, so 300+ LoRAs cost nothing until they matter
- Trigger words from the safetensors header, sidecars, or your own generation history
- Separates **accelerators** (turbo/distill/dmd2/lcm) from content LoRAs, because they change the sampling regime rather than the image

### 🔌 Environment probe ⭐
- Routes read from the live OpenAPI document, not a hardcoded list
- Reports which metadata sources exist and degrades when they do not
- Explains an empty history — including when `enable_pnginfo` and `save_txt` are both off

---

## 📦 Installation

### 1. Enable the API in Forge Neo

Add `--api` to your `COMMANDLINE_ARGS`:

```bat
set COMMANDLINE_ARGS=--api
```

That is the entire Forge-side change. Nothing is installed into your Forge folder.

### 2. Install the bridge

```bash
pip install git+https://github.com/eduardoabreu81/forgeneo-mcp
```

### 3. Register it with your agent

**Claude Code**

```bash
claude mcp add forgeneo -e FORGE_URL=http://127.0.0.1:7860 -- forgeneo-mcp
```

**Any client using `mcp.json`**

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

Requires Python 3.10+ and a Forge Neo instance reachable over HTTP.

### Forge on another machine

Start Forge with `--listen --api`, point `FORGE_URL` at it, and map its paths so results come back as files instead of base64:

```bash
claude mcp add forgeneo \
  -e FORGE_URL=http://gpu-box:7860 \
  -e FORGE_PATH_MAP='I:/sd-webui-forge-neo=//gpu-box/I/sd-webui-forge-neo' \
  -- forgeneo-mcp
```

> [!Note]
> `--listen` exposes the API to your network without authentication. Add `--api-auth user:password` and set `FORGE_AUTH` to match if that matters on your network.

---

## ⚙️ Configuration

Everything is optional except the URL when Forge is not on localhost.

| Variable | Purpose | Default |
|---|---|---|
| `FORGE_URL` | Base URL of the instance | `http://127.0.0.1:7860` |
| `FORGE_AUTH` | `user:password` when started with `--api-auth` | none |
| `FORGE_PATH_MAP` | `REMOTE=LOCAL` prefix pairs, `;`-separated | none |
| `FORGE_OUTPUT_DIR` | Output folder, when it cannot be derived | auto |
| `FORGE_TIMEOUT` | Request timeout in seconds | `600` |
| `FORGE_HISTORY_LIMIT` | How many recent outputs to index | `600` |
| `FORGE_CIVITAI_LOOKUP` | `1` allows an optional lineage lookup by file hash | off |
| `FORGENEO_CACHE_DIR` | Where confirmed dialects are cached | `~/.forgeneo-mcp` |

Without `FORGE_PATH_MAP` the bridge still works — it just returns base64 rather than paths.

`FORGE_CIVITAI_LOOKUP` is the **only** feature that leaves your machine: a public, read-only by-hash endpoint, no account, nothing uploaded but the hash Forge already computed.

---

## 🛠️ Tools

| Tool | What it answers |
|---|---|
| `capabilities` | What this instance offers, and which metadata sources exist |
| `model_profile` | What is loaded, how it behaves, and whether its modules are intact |
| `prompt_dialect` | How this checkpoint expects to be prompted, with its quality tags |
| `loras` | Which LoRAs match an intent, with trigger words and typical weights |
| `lora_info` | Full detail for one LoRA, with a ready prompt fragment |
| `models` | List, load or refresh checkpoints |
| `generate` | Generate from a written prompt — txt2img or img2img |
| `progress` | Check, interrupt or skip the running job |

---

## 🧠 How it decides

Each answer carries its provenance, and falls back honestly rather than guessing.

**Sampling parameters** — the architecture preset is authoritative for sampler, scheduler and shift; history adjusts only step count and guidance, which is what an accelerator LoRA actually changes. A regime belongs to a *(checkpoint, accelerators)* pair: ten steps on a base model is a property of the turbo LoRA loaded beside it, and the profile says so.

**Why parameters are sent explicitly** — leaving a field out of an API request does not fall back to the loaded architecture's value; it falls back to the API model's own default. Forge declares `distilled_cfg_scale = 3.5` and feeds it straight into `set_shift()`, so an omitted field silently generates at the wrong shift.

**Whether an architecture uses shift** — answered by observation, not a list. Forge writes `Shift` into infotext only for engines that consume it, so past generations settle it.

**Prompt dialect** — cache → optional CivitAI lookup → your own past prompts → architecture → the shape of your LoRA library. When all of that comes up short, it returns `unknown` with the candidate lineages instead of guessing, and your answer is cached by file hash so the question is asked once.

**Metadata sources** — the safetensors header, `.json` sidecars, generation history and the `.txt` infotext sibling are each optional and each a fallback for the others. A clean install simply reports less, and is told what is missing.

---

## 🗺️ Roadmap

- **Video (Wan)** — Forge generates video through `batch_size` in multiples of `4n+1` and encodes with ffmpeg, but the API discards the resulting `video_path`. Collecting from disk is already how images come back, so this is mostly plumbing.
- **EXIF metadata** — JPEG and WebP carry parameters in EXIF when `save_txt` is off; that configuration currently yields no history.
- **Authentication** — `FORGE_AUTH` is implemented but has not been exercised against a live `--api-auth` instance.

---

## 📄 Credits

- **[Forge Neo](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo)** by Haoming02 — the WebUI this bridges to
- **[Anima](https://huggingface.co/circlestone-labs)** by CircleStone Labs — model card that corrected the Anima dialect
- **[Model Context Protocol](https://modelcontextprotocol.io)** — the protocol and Python SDK
- **[CivitAI](https://civitai.com)** — public by-hash endpoint used by the optional lineage lookup

---

## 📜 License

MIT — see [LICENSE](LICENSE)

---

<div align="center">

Made with ❤️ for the Stable Diffusion community

**[Report Bug](https://github.com/eduardoabreu81/forgeneo-mcp/issues)** • **[Request Feature](https://github.com/eduardoabreu81/forgeneo-mcp/issues)** • **[Discussions](https://github.com/eduardoabreu81/forgeneo-mcp/discussions)** • **[☕ Ko-fi](https://ko-fi.com/eduardoabreu81)**

</div>
