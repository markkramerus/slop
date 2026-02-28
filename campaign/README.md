# Campaign Planner

A standalone application that decomposes a natural-language scenario into a structured **campaign plan** — a reviewable, editable JSON file that tells syncom how to distribute synthetic comments across argument angles, stakeholder types, and attack vectors.

## The Problem It Solves

Without the campaign planner, generating a diverse set of synthetic comments requires you to manually:
- Craft a precise `--objective` string
- Choose a single `--vector` (and run multiple times for a mix)
- Hope the LLM picks good argument angles on its own

With the campaign planner, you describe your scenario in natural language — your position, why you hold it, who would agree — and the planner produces a structured strategy that syncom executes.

## Workflow

```
┌─────────────────┐     ┌──────────────┐     ┌─────────────────┐
│  Scenario Brief  │────▶│   Planner    │────▶│ campaign_plan.  │
│  (your words)    │     │  (LLM call)  │     │     json        │
└─────────────────┘     └──────────────┘     └────────┬────────┘
                                                       │
                                              Human reviews &
                                              optionally edits
                                                       │
                                                       ▼
                                             ┌─────────────────┐
                                             │     syncom      │
                                             │   --campaign-   │
                                             │      plan       │
                                             └─────────────────┘
```

## Quick Start

### Step 1: Write a scenario brief

Create a text file (or just pass inline text) describing:
- What the rule proposes
- What your position is
- Why you hold that position
- What types of stakeholders would share your view

**Example** (`scenario_hti5.txt`):
```
HTI-5 proposes to fully remove the AI "model card" requirements from the
Decision Support Interventions (DSI) certification criterion. Today, developers
of certified health IT that include predictive DSIs must provide detailed source
attribute information — intended use, target population, known risks, training
data characteristics, and external validation processes. HTI-5 would eliminate
that disclosure requirement.

I strongly oppose this change. Removing AI transparency requirements in
healthcare puts patients at serious risk. Without model cards, clinicians have
no way to evaluate whether an AI tool is appropriate for their patient
population. Hospitals face liability exposure from deploying opaque AI systems.
And removing disclosure contradicts the federal government's own responsible AI
commitments.

Stakeholders who would share this position include: hospital associations,
clinical informatics organizations, patient safety advocacy groups,
cybersecurity-focused healthcare stakeholders, academic researchers in AI
ethics and health informatics, and individual clinicians who rely on AI tool
documentation to make informed decisions.
```

### Step 2: Run the planner

```bash
python campaign/planner.py \
  --rule-text HTI-5-Proposed-2025-23896.txt \
  --scenario scenario_hti5.txt \
  --output campaign_plan.json
```

### Step 3: Review and edit the plan

Open `campaign_plan.json` and review:
- **`objective`** — Is it the right framing?
- **`argument_angles`** — Are these the angles you want? Remove weak ones, add missing ones, adjust weights.
- **`stakeholder_emphasis`** — Does the archetype mix match your intent?
- **`vector_mix`** — Do you want more substance (vectors 1-3) or more volume (vector 4)?

### Step 4: Generate comments using the plan

```bash
python cli.py \
  --docket-id CMS-2025-0050-0031 \
  --rule-text HTI-5-Proposed-2025-23896.txt \
  --campaign-plan campaign_plan.json \
  --volume 50 \
  --output synthetic_comments.txt
```

When `--campaign-plan` is provided:
- `--objective` is read from the plan (no need to specify it)
- `--vector` is optional (the plan's `vector_mix` distributes across vectors)
- Comments are distributed across argument angles, stakeholder types, and vectors

## Campaign Plan Schema

```json
{
  "plan_version": "1.0",
  "created": "2026-02-27T15:13:00Z",
  "scenario_summary": "Brief summary of the position and goals",
  
  "objective": "The refined position all comments should advance",
  
  "argument_angles": [
    {
      "id": "patient_safety",
      "angle": "Removing AI transparency requirements creates patient safety risks",
      "weight": 0.25,
      "best_archetypes": ["advocacy_group", "individual_consumer"]
    }
  ],
  
  "stakeholder_emphasis": {
    "individual_consumer": 0.15,
    "advocacy_group": 0.30,
    "industry": 0.25,
    "academic": 0.20,
    "government": 0.10
  },
  
  "vector_mix": {
    "1": 0.35,
    "2": 0.40,
    "3": 0.15,
    "4": 0.10
  },
  
  "notes": "Strategic rationale from the planner"
}
```

### Field Reference

| Field | Description |
|---|---|
| `objective` | The position all comments should advance or oppose |
| `argument_angles[].id` | Short snake_case identifier |
| `argument_angles[].angle` | One-sentence description of this argument lens |
| `argument_angles[].weight` | Relative importance (normalized at runtime) |
| `argument_angles[].best_archetypes` | Which persona types best suit this angle |
| `stakeholder_emphasis` | Archetype → weight map for persona sampling bias |
| `vector_mix` | Attack vector → weight map for distributing comment styles |
| `notes` | Human-readable strategic rationale |

### Valid Archetypes

- `individual_consumer` — Patients, clinicians, caregivers, ordinary citizens
- `advocacy_group` — Patient safety orgs, professional societies, nonprofits
- `industry` — Hospital associations, health IT companies, health system execs
- `academic` — Researchers, professors, health policy scholars
- `government` — State/local health officials, Medicaid directors

### Attack Vectors

| Vector | Name | Description |
|---|---|---|
| 1 | Semantic Variance | Same argument, maximally varied surface forms |
| 2 | Persona Mimicry | Diverse stakeholders, same underlying position |
| 3 | Citation Flooding | Comments loaded with plausible-sounding references |
| 4 | Dilution/Noise | Brief, vague agreement (volume over substance) |

## Environment Variables

The planner uses the same API configuration as syncom (only the chat API — no embeddings needed):

| Variable | Default | Description |
|---|---|---|
| `SLOP_API_BASE_URL` | `https://api.openai.com/v1` | Chat API base URL |
| `SLOP_API_KEY` | (required) | Chat API key |
| `SLOP_CHAT_MODEL` | `gpt-4o` | Chat model name |

## CLI Reference

```
usage: campaign-planner --scenario PATH_OR_TEXT --rule-text PATH_OR_TEXT --output PATH

required arguments:
  --scenario PATH_OR_TEXT    Scenario brief file or inline text
  --rule-text PATH_OR_TEXT   Proposed rule text file or inline text
  --output PATH              Destination for campaign_plan.json

API configuration:
  --api-base-url URL         Override SLOP_API_BASE_URL
  --api-key KEY              Override SLOP_API_KEY
  --chat-model MODEL         Override SLOP_CHAT_MODEL

  --quiet                    Suppress progress output
```
