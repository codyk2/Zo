# Zo — EMPIRE

**A 24/7, multilingual, AI livestream-shopping seller — run end-to-end by Gemma 4 on Cactus, on a laptop.**

Built at the [Cactus × Google DeepMind × YC Gemma 4 Voice Agents hackathon](https://www.ycombinator.com/) (April 2026).

> Voice in → Gemma sees the product → Gemma writes the pitch → Gemma classifies every viewer comment → Gemma picks which pre-rendered clip to stitch in next. **~90% of comments never leave the laptop.** **$144/mo** unit economics vs. **~$9,628/mo** for an equivalent cloud + human team.

---

## Table of contents

- [The one-paragraph pitch](#the-one-paragraph-pitch)
- [Why this exists — the market gap](#why-this-exists--the-market-gap)
- [How Gemma 4 runs the entire seller](#how-gemma-4-runs-the-entire-seller)
  - [Gemma's five jobs](#gemmas-five-jobs)
  - [The three-tier stage](#the-three-tier-stage)
  - [One Gemma call → three parallel tier outputs](#one-gemma-call--three-parallel-tier-outputs)
  - [Fast path vs. slow path](#fast-path-vs-slow-path)
- [The cost case](#the-cost-case)
- [Repo map](#repo-map)
- [Getting started](#getting-started)
- [Phone as camera](#phone-as-camera)
- [Testing](#testing)
- [Shipped vs. out of scope](#shipped-vs-out-of-scope)
- [Further reading](#further-reading)
- [Credits](#credits)

---

## The one-paragraph pitch

Live commerce is a $68B US market (2026) converting at **10× traditional e-commerce**, but 95% of sellers are locked out because going live requires a human in every language, in every time zone, answering every comment — a payroll nobody except TikTok-native brands can afford. Replace the human with cloud AI and you trade a $10K/mo person for a $9.6K/mo API bill; the unit economics don't change. **EMPIRE runs the entire seller on-device** — Gemma 4 E4B via Cactus on the seller's MacBook handles pitch generation, comment classification, voice-intent parsing, local Whisper transcription, *and* draft response generation. The visible avatar is a three-tier library of pre-rendered MP4 clips that Gemma stitches in real time; only the ~10% of novel responses get a mouth-region lip-sync on a RunPod 5090. End-to-end: **~$144/mo per seller, 140+ languages, 24/7, every comment answered.**

---

## Why this exists — the market gap

Live commerce is selling products through livestreams: a host shows the item, talks about it, answers questions in chat, and viewers buy directly from the stream. QVC for the TikTok generation, run by individual sellers instead of TV networks.

| Metric | Number |
|---|---|
| US live commerce market (2026) | **$68 billion** |
| Global live commerce (China) | **$500 billion** |
| TikTok Shop global sales (last quarter) | **$19 billion** |
| US TikTok Shop YoY growth | **+125%** |
| Live commerce conversion rate | **30%** (vs. 2–3% for static e-commerce — **10× the rate**) |

Going live is brutal — it's the reason 95% of sellers can't participate:

| Problem | What it actually costs |
|---|---|
| **Languages** | A human host speaks 1–2. Your customers are in 140+. |
| **Time zones** | A human host can do ~8 hours/day. Your buyers are awake 24/7 — you miss two-thirds of the world. |
| **Team cost** | Solo host: $5,160/mo. Agency: **$8,000–$12,500/mo**. Plus camera, lighting, producer, comment moderator, sales coach. |
| **Cloud-AI alternative** | If you naively replace the team with cloud LLM + cloud TTS + cloud avatar, you trade a $10K/mo human bill for a **$9,628/mo cloud bill**. The unit economics still don't work. |
| **EMPIRE** | **$144/mo, 140+ languages, 24/7, every comment answered.** |

That's the wedge. The only way the unit economics close is if the on-device model handles ~90% of the work. **That model is Gemma 4 on Cactus.**

---

## How Gemma 4 runs the entire seller

Gemma 4 E4B (via the [Cactus](https://cactuscompute.com/) runtime on Apple Silicon) is the brain of every decision EMPIRE makes. Not a fallback, not an accelerator — the **primary intelligence layer**. Every other component (Wav2Lip, ElevenLabs, the pre-rendered clip library, the dashboard) is plumbing that Gemma drives.

> **Today's hardware state:** Gemma is running on CPU prefill via Cactus (2–4 s per classify). The `model.mlpackage` (Apple Neural Engine weights) is the pending optimization with a <500 ms target. See [`phase0/PHASE0_REPORT.md`](./phase0/PHASE0_REPORT.md) for the honest latency budget.

### Gemma's five jobs

All five run **on the seller's laptop**. Zero cloud spend, zero IP leak, zero latency to a datacenter.

| # | Gemma job | Code | What it does |
|---|---|---|---|
| **1** | **Pitch generation (vision + script)** | [`backend/agents/eyes.py`](./backend/agents/eyes.py) — `analyze_and_script_gemma` (fused single call) | Looks at the product photo + the seller's narration. Returns one JSON: `{product:{name, materials, selling_points, visual_details}, script:"<70-word TikTok-energy pitch>"}`. **Replaces a Claude vision call.** Falls back to Claude only if Gemma's JSON is malformed. |
| **2** | **Comment classification (intent routing)** | [`backend/agents/eyes.py`](./backend/agents/eyes.py) — `classify_comment_gemma` | Every viewer comment → `{type: question\|compliment\|objection\|spam, draft_response}`. This is the input the router uses to pick which pre-rendered clip plays next. |
| **3** | **Voice intent parsing** | [`backend/agents/eyes.py`](./backend/agents/eyes.py) — `parse_voice_intent_gemma` | The seller says *"sell this for $89, target young professionals."* Gemma extracts `{action:"sell", price:89, target_audience:"young professionals"}` — structured, no regex. |
| **4** | **On-device Whisper transcription** | [`backend/agents/eyes.py`](./backend/agents/eyes.py) — `_cactus_transcribe_audio` | Whisper-base via Cactus, **244 ms observed** for a ~1.2 s utterance. Push-to-talk mic → text without a Gemini API round-trip. |
| **5** | **Drafting the comment response itself** | Same `classify_comment_gemma` call returns a `draft_response` field that becomes the spoken audio in the dispatch path | When the local Q&A index doesn't match, Gemma's draft_response is what gets TTS'd and lip-synced onto the next pre-rendered substrate clip. **No second LLM call** — one Gemma forward pass produces both the routing decision *and* the words the avatar will say. |

**Without Gemma, EMPIRE doesn't have a product.** The cloud fallback chain (Claude / Gemini Speech) exists only as a safety net for the demo — every shipped path runs Gemma first.

### The three-tier stage

EMPIRE does **not** generate video in real time. What looks to the audience like a live, reactive AI avatar is actually a **library of pre-rendered MP4 clips that Gemma 4 intelligently stitches together**.

The dashboard composites **three tiers of pre-rendered MP4** onto the screen at all times. Gemma 4 is the router that decides which tier fires, when, and what audio rides on top.

```
                ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
                ┃        G E M M A   4   E 4 B           ┃
                ┃        ─────────────────────           ┃
                ┃        on-device · <500 ms target      ┃
                ┗━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┛
                                     │
                                     │  routes EVERY state
                                     │  transition on stage
                                     ▼
                            ┌─────────────────┐
                            │    DIRECTOR     │   choreographer FSM
                            │  (per-process)  │   (backend/agents/
                            └────────┬────────┘    avatar_director.py)
                                     │
        ┌────────────────────────────┼────────────────────────────┐
        │                            │                            │
        ▼                            ▼                            ▼
  ┌───────────┐                ┌───────────┐                ┌───────────┐
  │  TIER 0   │                │  TIER 1   │                │  TIER 2   │
  │  ───────  │                │  ───────  │                │  ───────  │
  │   IDLE    │  ◀── Gemma ──  │  BRIDGE   │  ◀── Gemma ──  │ RESPONSE  │
  │           │   swaps pose   │           │   picks bucket │           │
  │ always on │                │ comment-  │                │ the actual│
  │ silent    │                │ intent    │                │   answer  │
  │ Veo loop  │                │ substrate │                │           │
  └───────────┘                └───────────┘                └───────────┘
```

| Tier | Content | When it fires | Gemma's role |
|---|---|---|---|
| **0 — Idle** | Looping muted Veo render of the avatar's resting pose. Never goes black. | Continuously. Every other tier crossfades on top of this. | Gemma's `voice_state` signal swaps the active idle pose (calm ↔ thinking) when the operator pauses to think. |
| **1 — Bridge** | 8-second pre-rendered intent substrate matching the comment's emotional register (warm-smile / thoughtful-nod / "actually here's the deal"). | The moment a comment lands. Crossfades onto Tier 0. | **Gemma's `classify_comment` returns `type` — that field IS the bucket selector.** The Director picks a random clip from the matching bucket. |
| **2 — Response** | Either a fully pre-rendered Q&A MP4 (when Gemma's intent matches a known question) or a fresh Wav2Lip lip-sync onto the Tier 1 substrate (when Gemma drafts a novel response). | Once TTS + Wav2Lip finishes. Crossfades onto Tier 1. | Gemma's `draft_response` is the audio source. Gemma's `qa_index` match decides whether a fresh render is even needed. |

**The "live AI avatar" is Gemma 4 driving three pre-rendered tiers in sequence.** That's it.

### One Gemma call → three parallel tier outputs

The trick isn't the model — it's the **prompt schema**. Gemma is constrained to emit one JSON object whose fields map 1:1 to downstream subsystems, so one forward pass produces multiple concurrent UI actions.

```
   ┌─────────────────────────────────────────────────────────────────┐
   │   GEMMA.classify_comment("is it real leather")                  │
   │   ────────────────────────────────────────────                  │
   │                                                                 │
   │   returns ONE JSON object:                                      │
   │     {                                                           │
   │       type:           "question",                               │
   │       draft_response: "Yes, full-grain vegetable-tanned         │
   │                        leather. See the natural grain?",        │
   │       latency_ms:     147                                       │
   │     }                                                           │
   └─────────────────────────────────┬───────────────────────────────┘
                                     │
              ┌──────────────────────┴──────────────────────┐
              │                                             │
              ▼                                             ▼
    TIER 1 BUCKET PICK                            TIER 2 AUDIO + RENDER
    ──────────────────                            ─────────────────────
    uses Gemma's `type` field                     uses Gemma's `draft_response`
    → indexes the matching                        → ElevenLabs TTS streams the
      intent bucket (compliment /                   text (translated to one of
      question / objection)                         6 supported languages on
    → Director picks 1 random                       cache miss)
      pre-rendered substrate                      → Wav2Lip overlays mouth
                                                    pixels onto the Tier 1
                                                    substrate the Director
                                                    just chose. Body language
                                                    is inherited; only the
                                                    mouth region is generated
                                                    live.

   ┌─────────────────────────────────────────────────────────────────┐
   │  (concurrent third channel — same Gemma model, different prompt)│
   │                                                                 │
   │  GEMMA.parse_voice_intent(seller_audio) ──►  drives TIER 0 pose │
   │     voice_state = "thinking" for >2 s  →  Tier 0 crossfades to  │
   │     idle_thinking pose; reverts on next state change.           │
   └─────────────────────────────────────────────────────────────────┘
```

No second LLM call. No stacked inference. The comment in [`backend/agents/eyes.py:389`](./backend/agents/eyes.py) explicitly claims "~35-45% faster than split when the model produces parseable JSON" — fusing isn't theoretical, it's measured.

### Fast path vs. slow path

**Path A — Gemma matches the seller's pre-authored Q&A index (~90% of comments)**

```
   TIME →     0 ms       150 ms      200 ms              600 ms

   TIER 0  ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
            idle_calm loop (always painting underneath)

   TIER 1                                  (skipped — fast path)

   TIER 2                          [████████ pre-rendered Q&A MP4 ████████]

   Total: < 600 ms.  Zero cloud spend.  Zero render.
   The MP4 was rendered once, at product onboarding, by the same
   Wav2Lip + ElevenLabs pipeline used live.
```

**Path B — Gemma drafts a novel response, Wav2Lip stitches it onto a Tier 1 substrate (~10%)**

```
   TIME →   0 s     0.2 s     0.5 s              2.0 s              7.0 s

   TIER 0  ▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
            idle_calm loop (always painting underneath the whole time)

   TIER 1                  [▓▓▓ thoughtful-nod bridge substrate ▓▓▓]

   TIER 2                                              [████ Wav2Lip lip-sync ████]
                                                        ▲
                                                        │
                                                        Audio = Gemma's draft_response
                                                          via ElevenLabs TTS.
                                                        Body language = the Tier 1
                                                          pose, preserved.
                                                        Mouth pixels = the only
                                                          thing rendered live.
```

Three crossfades. One Gemma forward pass. Zero generative video. That's the whole trick.

---

## The cost case

| Approach | Monthly cost (modeled) | Languages | Hours/day | Comment coverage |
|---|---|---|---|---|
| Hire a human team | **$5,160 – $12,500** | 1–2 | 8 | Misses 60–80% during busy streams |
| 100% cloud AI (Claude/GPT for everything) | **~$9,628** | 10–20 | 24 | Every one |
| **EMPIRE (Gemma 4 + Cactus on-device, cloud only on escalation)** | **~$144** | **140+** | **24** | **Every one** |

The on-device leg — Cactus + Gemma 4 voice + classify + Q&A index + draft_response + the pre-rendered clip library — costs **$0** in marginal spend. The only cloud bills are:

- ~$60 ElevenLabs (TTS for the ~10% that escalate + bridge audio)
- ~$25 Bedrock Claude Haiku (cloud responses for the 10% Gemma defers)
- ~$50 RunPod (5090 spot, ~3 active hours/day)
- ~$9 Anthropic spot for product analysis fallback

**This isn't a tech-flex.** It's the only architecture where 24/7 multilingual AI live commerce is economically viable. Without Gemma 4 on-device, the unit economics are the same as a human team — and you've added an AI startup's complexity tax on top.

Full model in [`EMPIRE-COST-ANALYSIS.md`](./EMPIRE-COST-ANALYSIS.md).

---

## Repo map

```
Zo/
├── backend/                 # FastAPI + agent pipeline (Python, uvicorn, port 8000)
│   ├── agents/
│   │   ├── eyes.py            # Gemma/Cactus wrapper — all 5 Gemma jobs live here
│   │   ├── router.py          # Rule-based comment router, 4-tool dispatcher
│   │   ├── avatar_director.py # Three-tier FSM — which clip plays when
│   │   ├── brain.py           # Telemetry: local-vs-cloud routing rate + cost saved
│   │   ├── bridge_clips.py    # Tier 1 intent substrate library indexer
│   │   ├── seller.py          # Pitch generation orchestrator
│   │   └── threed.py          # 3D spin carousel asset pipeline
│   ├── main.py                # API surface, WebSocket broadcasts
│   ├── phone_uploader.py      # Integrated phone-capture endpoint
│   ├── static/
│   │   ├── phone_recorder.html  # Integrated phone UI (served on /phone)
│   │   └── debug_clips.html     # Dev-only clip inspector
│   ├── data/products.json     # Products + their qa_index
│   └── tests/                 # pytest suite (17 router cases, endpoint smoke)
│
├── dashboard/               # React + Vite operator UI (port 5173)
│   └── src/
│       ├── App.jsx
│       ├── components/
│       │   ├── LiveStage.jsx       # Three-tier MP4 compositor on screen
│       │   ├── ChatPanel.jsx       # Simulated viewer comments feed
│       │   ├── BrainPanel.jsx      # Live local/cloud/cost telemetry
│       │   ├── Spin3D.jsx          # Product 3D preview
│       │   ├── PhoneQRPanel.jsx    # QR code for phone-as-camera pairing
│       │   ├── EventLogHUD.jsx     # Debug event firehose
│       │   └── ProductSelector.jsx
│       └── hooks/
│           ├── useEmpireSocket.js  # WS client to backend
│           └── useAvatarStream.js  # Tier-0/1/2 state subscription
│
├── phone-quickdrop/         # Standalone lightweight phone→laptop video drop
│   ├── server.js              # HTTPS + WebSocket server (port 8443)
│   ├── public/phone.html      # Phone recorder UI
│   ├── public/desktop.html    # Laptop dashboard (QR + live recordings)
│   ├── scripts/smoke.js       # End-to-end smoke test
│   └── scripts/smoke-reconnect.js  # Reconnect-recovery regression test
│
├── ios/                     # Standalone iPhone demo (Swift, Airplane-Mode ready)
│   └── EmpirePhone/           # Cactus whisper + port of the rule-based router
│
├── mobile/                  # React Native / Expo mobile app (experimental)
│
├── phase0/                  # Hackathon Phase 0 artifacts + fixtures
│   ├── assets/bridges/        # Tier 1 intent substrate MP4s
│   ├── scripts/               # Pod provisioning, bridge upload, cert helpers
│   ├── fixtures/              # Demo comment sets, CODY_PROMPT.md
│   └── PHASE0_REPORT.md       # Honest latency + scope retro
│
├── runpod/                  # Wav2Lip server, Dockerfile, provisioning
├── scripts/                 # Top-level utilities (render_local_answers.py, etc.)
├── docs/                    # Design-principles PDFs, supplementary docs
│
├── README.md                # (this file)
├── ARCHITECTURE.md          # Full build — every agent, every tier, every escalation
├── EMPIRE-PITCH.md          # Thesis deck in prose form
├── EMPIRE-COST-ANALYSIS.md  # Line-by-line unit economics
├── DESIGN_PRINCIPLES.md     # Lidwell index for UI reasoning
├── PHONE.md                 # Phone-side architecture deep-dive
├── BUILD_NOTES.md           # Chronological build log
├── SHIP.md                  # Pre-launch checklist
├── STAGE_RUNBOOK.md         # Live-demo operator playbook
└── Makefile                 # make dev / make test / make demo
```

---

## Getting started

### Prerequisites

- **macOS with Apple Silicon** (M1 or newer)
- [Cactus](https://cactuscompute.com/) + Gemma 4 E4B + whisper-base downloaded at `~/cactus/`
- RunPod pod with Wav2Lip server on port 8010 (see [`scripts/provision_pod.sh`](./scripts/provision_pod.sh))
- `.env` with `ELEVENLABS_API_KEY`, `ELEVENLABS_VOICE_ID`, `AWS_*`, `RUNPOD_*` — template in [`.env.example`](./.env.example)
- Python 3.11+, Node 18+

### Boot

```bash
# 1. Open SSH tunnel to the Wav2Lip pod (keeps running)
bash phase0/scripts/open_tunnel.sh

# 2. Regenerate pre-rendered local answers on fresh clones (see below)

# 3. Backend
cd backend && uvicorn main:app --host 0.0.0.0 --port 8000

# 4. Dashboard
cd dashboard && npm install && npm run dev
# → open http://localhost:5173
```

### Regenerating the pre-rendered local-answer library

`backend/local_answers/*.mp4` is gitignored (~50 MB of pre-rendered MP4 per product). A fresh clone has no clips, so every local-index question falls through to the cloud-escalate path. To regenerate:

```bash
# 1. Confirm ELEVENLABS_API_KEY + ELEVENLABS_VOICE_ID are set in .env
# 2. Confirm the Wav2Lip tunnel is up: curl http://127.0.0.1:8010/health
# 3. Run the pre-render pipeline (10 clips, ~46 s wall-time):
python scripts/render_local_answers.py
# Use --force to re-render after changing ELEVENLABS_VOICE_ID.
```

The bridge clips (compliment / question / objection intent substrates) are uploaded to the pod separately:

```bash
bash phase0/scripts/upload_bridge_clips.sh
```

---

## Phone as camera

Two separate phone-capture flows exist in the repo — keep straight which one you want.

### 1. Integrated capture — `backend/phone_uploader.py` + `dashboard/components/PhoneQRPanel.jsx`

The dashboard shows a `PhoneQRPanel`. Scan the QR on your phone → it opens a recorder page served by the backend. The recorded video uploads back to the backend and feeds directly into the seller pipeline (product analysis, pitch generation). **Use this when recording a product to list.**

### 2. Standalone quickdrop — `phone-quickdrop/` (port 8443)

Lightweight phone → laptop video drop with no backend dependency. Phone records video; file lands in `~/Desktop/PhoneCaptures/` the instant you tap Stop. Streams chunks via WebSocket every 250 ms (vs. AirDrop which handshakes after you hit Send — quickdrop is effectively instant on LAN). Useful for grabbing clips for the bridge library or ad-hoc captures.

```bash
cd phone-quickdrop
npm install
npm start
# → open https://localhost:8443/ on laptop, scan the QR with your phone
```

**Networking gotcha:** both flows require the phone to route to the laptop's LAN IP. Guest Wi-Fi (coworking, hotels, YC) typically has **AP client isolation** which blocks peer-to-peer traffic and makes this fail silently. Workaround: USB tether — plug iPhone into Mac, enable Personal Hotspot, Mac auto-joins the hotspot net. Zero cellular data burned (phone↔laptop traffic stays on the USB link).

---

## Testing

```bash
# Router unit tests (19 parametrized cases covering question/compliment/objection/spam,
# qa_index matches, cloud escalation, bridge bucket picks)
pytest backend/tests/test_router.py -v

# Endpoint smoke (API surface + WebSocket handshake)
pytest backend/tests/test_endpoints_smoke.py -v

# Phone-quickdrop end-to-end (send 1 MiB synthetic video, verify disk bytes)
cd phone-quickdrop && npm run smoke

# Reconnect-recovery regression (kill WS mid-upload, verify file reassembles intact)
cd phone-quickdrop && npm run smoke:reconnect
```

Live cURL probes against a running backend:

```bash
# Fast path — Gemma classifies + matches Q&A + plays clip in <1s
curl -F "text=is it real leather" http://127.0.0.1:8000/api/comment

# Slow path — Gemma classifies + drafts + Wav2Lip stitches onto pre-rendered substrate
curl -F "text=how does this compare to the Apple Watch" http://127.0.0.1:8000/api/comment
```

---

## Shipped vs. out of scope

**Built and shipped:**

- Gemma-driven pitch generation (vision + script in one fused call)
- Gemma comment classifier + 4-tool router (17 unit tests passing)
- On-device Whisper transcription via Cactus (~244 ms observed)
- Three-tier Director with pre-rendered clip choreography (Tier 0 idle / Tier 1 intent bridge / Tier 2 response)
- Wav2Lip lip-sync overlay onto intent-aware Tier 1 substrates
- Pre-rendered local Q&A library (10–20 MP4s per product at ~50 MB)
- Multilingual TTS path (6 languages today, architecture supports 140+)
- BRAIN telemetry panel showing local-vs-cloud routing rate + cost saved
- Phone-as-camera capture (both integrated and standalone flows)
- Standalone iOS EmpirePhone demo with Airplane-Mode-capable router

**Out of scope for the demo:**

- Real TikTok Shop / Instagram Live integration. The chat panel simulates comments; the router itself is identical to what would feed a real platform integration.
- CLOSER agent (DM auto-responder)
- Generative photo variants
- Multi-stream isolation

Full honest list in [`ARCHITECTURE.md`](./ARCHITECTURE.md) §"Not in scope today".

---

## Further reading

| Doc | Purpose |
|---|---|
| [`ARCHITECTURE.md`](./ARCHITECTURE.md) | Every agent, every tier, every escalation path, with file:line refs |
| [`EMPIRE-PITCH.md`](./EMPIRE-PITCH.md) | The thesis in prose form — market gap, wedge, moat |
| [`EMPIRE-COST-ANALYSIS.md`](./EMPIRE-COST-ANALYSIS.md) | Line-by-line unit economics |
| [`DESIGN_PRINCIPLES.md`](./DESIGN_PRINCIPLES.md) | Lidwell indexed for agents reasoning about UI decisions |
| [`PHONE.md`](./PHONE.md) | Phone-side architecture deep-dive |
| [`STAGE_RUNBOOK.md`](./STAGE_RUNBOOK.md) | Live-demo operator playbook |
| [`phase0/PHASE0_REPORT.md`](./phase0/PHASE0_REPORT.md) | Honest latency + scope retro |
| [`ios/README.md`](./ios/README.md) | Standalone iPhone demo scope |

---

## Credits

Built at the **Cactus × Google DeepMind × YC Gemma 4 Voice Agents Hackathon**, April 2026.

Stack:

- **[Gemma 4 E4B](https://ai.google.dev/gemma)** — on-device intelligence layer
- **[Cactus](https://cactuscompute.com/)** — M-series runtime for Gemma + Whisper
- **[Wav2Lip](https://github.com/Rudrabha/Wav2Lip)** — mouth-region overlay on pre-rendered substrates
- **[ElevenLabs](https://elevenlabs.io/)** — multilingual TTS on the ~10% escalate path
- **[Veo](https://deepmind.google/technologies/veo/)** — offline pre-rendering of Tier 0 idle loops and Tier 1 intent substrates
- **[LatentSync 1.6](https://github.com/bytedance/LatentSync)** — per-product pitch-clip renders
- **[RunPod](https://runpod.io/)** — 5090 pod hosting the live lip-sync server
