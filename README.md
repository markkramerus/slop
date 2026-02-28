# SLOP - Synthetic Letter-writing Opposition Platform

## Quick Start

### Generate synthetic public comments in 5 steps:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set up your API keys
# Create a .env file with:
#   SLOP_API_KEY=your_key_here
#   SLOP_EMBED_API_KEY=your_key_here

# 3. Download a docket and convert attachments to text
python downloader/download_attachments.py CMS-2025-0050 --convert-text

# 4. Analyze writing styles in the docket
python stylometry/stylometry_analyzer.py CMS-2025-0050

# 5. Generate synthetic comments
python cli.py \
    --docket-id CMS-2025-0050 \
    --vector    2 \
    --objective "oppose the proposed reduction of Medicare Advantage quality bonus payments" \
    --volume    10
```

That's it! You'll have 10 synthetic public comments written in the voice styles of real commenters from that docket, saved to `CMS-2025-0050/synthetic_comments/synthetic.txt`.

**Even simpler with a campaign plan** — once you have a `CMS-2025-0050/campaign/campaign_plan.json`, the only required arguments are `--docket-id` and `--volume`:

```bash
python cli.py --docket-id CMS-2025-0050 --volume 50
```

## What Does This Do?

SLOP generates **realistic synthetic public comments** for regulatory dockets. Think of it as creating AI-generated letters that could plausibly have been written by real stakeholders—healthcare providers, patients, advocacy groups, industry representatives, etc.

### Example Input
A real public comment from regulations.gov:
```
"Our clinic in Minnesota uses Medicare Advantage quality bonuses to fund 
infrastructure upgrades. Reducing these bonuses will force us to cut 
corners on health IT..."
```

### Example Output
A synthetic comment with similar themes but different voice and details:
```
"As a practice manager in Ohio, I've seen firsthand how quality bonuses 
enable small clinics to adopt interoperable EHR systems. Without this 
funding, we'll be locked into legacy systems..."
```

## Why Would You Use This?

1. **Stress-test analysis pipelines** - Test your comment analysis tools with diverse inputs
2. **Research regulatory capture** - Understand how sophisticated comments differ from grassroots input
3. **Train comment classification models** - Generate labeled training data
4. **Simulate public engagement** - Model what various stakeholder groups might say
5. **Academic research** - Study computational persuasion and astroturfing

## Installation

### Prerequisites
- Python 3.8 or higher
- OpenAI-compatible API key (GPT-4o recommended; any OpenAI-compatible endpoint works)

### Install

```bash
# Clone the repository
git clone https://github.com/markkramerus/slop.git
cd slop

# Install dependencies
pip install -r requirements.txt

# Set up your API keys
echo "SLOP_API_KEY=your_key_here" >> .env
echo "SLOP_EMBED_API_KEY=your_key_here" >> .env
```

### Environment Variables

| Variable | Description |
|---|---|
| `SLOP_API_BASE_URL` | Chat API base URL (default: `https://api.openai.com/v1`) |
| `SLOP_API_KEY` | Chat/completion API key (required) |
| `SLOP_CHAT_MODEL` | Chat model name (default: `gpt-4o`) |

## Sub-Applications

SLOP is organized into five sub-applications, each with its own README:

| Sub-Application | Directory | Purpose |
|---|---|---|
| **Downloader** | `downloader/` | Download attachments from regulations.gov dockets |
| **Stylometry** | `stylometry/` | Analyze real-comment writing styles; generate voice skill files |
| **Campaign Planner** | `campaign/` | Decompose a scenario into a structured generation strategy |
| **syncom** | `syncom/` | Core synthetic comment generator |
| **Shuffler** | `shuffler/` | Translate synthetic output to CMS CSV format and shuffle it into a real comment file with a ground-truth key |

---

### Downloader (`downloader/`)

Downloads attachment files from regulations.gov CSV exports and organizes them in a docket-centric directory hierarchy.

```bash
# Download all attachments for a docket (docket ID → derives CSV path automatically)
python downloader/download_attachments.py CMS-2025-0050

# Download and convert PDF/DOCX to .txt for downstream processing
python downloader/download_attachments.py CMS-2025-0050 --convert-text

# Explicit CSV path also still works
python downloader/download_attachments.py CMS-2025-0050/comments/CMS-2025-0050.csv --convert-text

# Convert already-downloaded files to text
python downloader/text_converter.py CMS-2025-0050
```

Output structure:
```
CMS-2025-0050/
└── comment_attachments/
    ├── CMS-2025-0050-0004/
    │   └── attachment_1.pdf
    └── CMS-2025-0050-0005/
        ├── attachment_1.pdf
        └── attachment_1.txt   ← converted text version
```

📖 See [`downloader/README_DOWNLOADER.md`](downloader/README_DOWNLOADER.md) for full documentation.

---

### Stylometry Analyzer (`stylometry/`)

Analyzes writing styles in real docket comments and generates reusable **voice skill** markdown files. These skills are used by the generator to produce synthetic comments that match the actual writing patterns of real commenters in that docket—rather than relying on generic instructions.

```bash
# Analyze a docket (docket ID → derives CSV path automatically)
python stylometry/stylometry_analyzer.py CMS-2025-0050

# Explicit CSV path also still works
python stylometry/stylometry_analyzer.py CMS-2025-0050/comments/CMS-2025-0050.csv
```

Output:
```
CMS-2025-0050/
└── stylometry/
    ├── index.json
    ├── individual_consumer-low.md
    ├── individual_consumer-medium.md
    ├── individual_consumer-high.md
    ├── industry-medium-org.md
    ├── industry-high-org.md
    ├── advocacy_group-medium-org.md
    └── ...
```

Voice groups are defined by archetype × sophistication level (e.g., `individual_consumer-high`, `industry-medium-org`). Each skill file contains empirical statistics (word count ranges, sentence lengths, punctuation frequencies) and AI-avoidance rules tailored to what that specific group of real commenters actually writes.

**Run this before generating comments.** The generator looks for `{docket_id}/stylometry/` automatically.

📖 See [`stylometry/STYLOMETRY_README.md`](stylometry/STYLOMETRY_README.md) for full documentation.

---

### Campaign Planner (`campaign/`)

Translates a natural-language scenario brief into a structured **campaign plan** — a reviewable, editable JSON file that tells the generator how to distribute synthetic comments across argument angles, stakeholder types, and attack vectors.

```
┌─────────────────┐     ┌──────────────┐     ┌──────────────────┐
│  Scenario Brief  │────▶│   Planner    │────▶│ campaign_plan.   │
│  (your words)    │     │  (LLM call)  │     │     json         │
└─────────────────┘     └──────────────┘     └────────┬─────────┘
                                                       │
                                              Human reviews &
                                              optionally edits
                                                       │
                                                       ▼
                                             ┌─────────────────┐
                                             │  cli.py         │
                                             │  --campaign-plan │
                                             └─────────────────┘
```

```bash
# Step 1: Generate a campaign plan (all paths derived from docket ID)
python campaign/planner.py --docket-id CMS-2025-0050

# Step 2: Review and edit CMS-2025-0050/campaign/campaign_plan.json, then generate
# campaign_plan.json is auto-detected — only --docket-id and --volume needed:
python cli.py --docket-id CMS-2025-0050 --volume 50

# Explicit paths still work if you want to override defaults:
python campaign/planner.py \
    --docket-id CMS-2025-0050 \
    --rule-text CMS-2025-0050/rule/rule.txt \
    --scenario  CMS-2025-0050/campaign/scenario_brief.txt \
    --output    CMS-2025-0050/campaign/campaign_plan.json
```

When `--campaign-plan` is provided, `--objective` and `--vector` are read from the plan; comments are automatically distributed across argument angles, stakeholder types, and vectors according to the plan's weights.

📖 See [`campaign/README.md`](campaign/README.md) for full documentation including the plan schema.

---

### syncom — Synthetic Comment Engine (`syncom/`)

The core generation engine. Invoked via `cli.py`.

**Attack Vectors:**

| Vector | Name | Description |
|---|---|---|
| 1 | Semantic Variance | Same argument, maximally varied surface forms |
| 2 | Persona Mimicry | Diverse stakeholders, same underlying position |
| 3 | Citation Flooding | Comments loaded with plausible-sounding references |
| 4 | Dilution / Noise | High-volume, low-substance vague agreement |

---

### Shuffler (`shuffler/`)

The final phase of the pipeline. It takes the ♔-delimited syncom output, translates it to standard CMS CSV format, then randomly interleaves the synthetic comments into a real CMS comment file. The result is a combined CSV indistinguishable in format from a real docket export, plus a **key file** that records the ground truth for every row.

#### Full pipeline via `cli.py shuffle`

```bash
# Simplest form — all paths derived from docket ID:
python cli.py shuffle --docket-id CMS-2025-0050

# Explicit paths also still work:
python cli.py shuffle \
    --syncom-output     CMS-2025-0050/synthetic_comments/synthetic.txt \
    --translated-output CMS-2025-0050/shuffled_comments/synthetic_cms.csv \
    --real-comments     CMS-2025-0050/comments/CMS-2025-0050.csv \
    --combined-output   CMS-2025-0050/shuffled_comments/combined.csv
```

This single command:
1. Translates `synthetic.txt` (♔-delimited) → `synthetic_cms.csv` (CMS CSV)
2. Loads the real comment file and the translated synthetic file
3. Randomly shuffles them together (reproducible via `--seed`)
4. Writes the combined file to `combined.csv`
5. Auto-generates `combined_key.csv` in the same directory

#### Key file format

The key CSV has three columns and one row per comment in the combined file:

| Column | Description |
|---|---|
| `row_number` | 1-based position in the combined CSV (not counting the header) |
| `document_id` | The `Document ID` value from that row |
| `type` | `real` or `synthetic` |

#### Skip the translation step

If you have already translated the synthetic comments (e.g., in a prior run), use `--skip-translation` to go straight to shuffling:

```bash
python cli.py shuffle --docket-id CMS-2025-0050 --skip-translation
```

#### Translation only (lower-level)

```bash
# Translate ♔-delimited → CMS CSV without shuffling
python shuffler/translate_to_cms_format.py \
    CMS-2025-0050/synthetic_comments/comments.txt \
    CMS-2025-0050/synthetic_comments/comments_cms.csv

# Verify the translation
python shuffler/verify_translation.py CMS-2025-0050/synthetic_comments/comments_cms.csv
```

📖 See [`shuffler/TRANSLATION_README.md`](shuffler/TRANSLATION_README.md) for field-mapping details.

---

## Full Workflow

All steps accept a docket ID and derive conventional file paths automatically.

```
1. Download docket
   python downloader/download_attachments.py {docket_id} --convert-text
   └── {docket_id}/comment_attachments/

2. Analyze writing styles
   python stylometry/stylometry_analyzer.py {docket_id}
   └── {docket_id}/stylometry/*.md

3a. Direct generation (simple)
    python cli.py --docket-id {docket_id} --vector N --objective "..." --volume N
    # rule text read from {docket_id}/rule/rule.txt
    # output written to {docket_id}/synthetic_comments/synthetic.txt

    OR

3b. Campaign-planned generation (structured)
    python campaign/planner.py --docket-id {docket_id}
    # reads  {docket_id}/rule/rule.txt + {docket_id}/campaign/scenario_brief.txt
    # writes {docket_id}/campaign/campaign_plan.json
    # Review and edit campaign_plan.json, then:
    python cli.py --docket-id {docket_id} --volume N
    # campaign_plan.json is auto-detected; output → {docket_id}/synthetic_comments/synthetic.txt

4. Translate + shuffle
   python cli.py shuffle --docket-id {docket_id}
   └── {docket_id}/shuffled_comments/
       ├── synthetic_cms.csv   ← translated synthetic comments in CMS format
       ├── combined.csv        ← real + synthetic, randomly interleaved
       └── combined_key.csv    ← ground-truth label for every row
```

## Command-Line Reference

### `cli.py` — Main Generator

```
python cli.py [OPTIONS]

Required:
  --docket-id ID        Docket identifier (e.g., 'CMS-2025-0050').
                        The tool looks for stylometry data in {docket_id}/stylometry/.
  --volume N            Number of accepted synthetic comments to produce.

Convention-based defaults (derived from --docket-id):
  --rule-text PATH      Default: {docket_id}/rule/rule.txt
  --output PATH         Default: {docket_id}/synthetic_comments/synthetic.txt
  --campaign-plan PATH  Auto-detected: {docket_id}/campaign/campaign_plan.json (if present)

Direct mode (required when no campaign plan is found):
  --objective TEXT      The position to advance or oppose.
  --vector {1,2,3,4}    Attack vector (1=Semantic Variance, 2=Persona Mimicry,
                        3=Citation Flooding, 4=Dilution/Noise).

Campaign plan mode:
  --campaign-plan PATH  Path to campaign_plan.json from campaign/planner.py.
                        Provides --objective and distributes across vectors.

API configuration:
  --api-base-url URL    Override SLOP_API_BASE_URL
  --api-key KEY         Override SLOP_API_KEY
  --chat-model MODEL    Override SLOP_CHAT_MODEL

Quality control:
  --no-relevance-check  Skip LLM topical-relevance QC
  --no-argument-check   Skip LLM argument-presence QC
  --no-embedding-check  Skip embedding-based deduplication
  --include-failed-qc   Write QC-failed rows to output (flagged)
  --similarity-threshold FLOAT  Cosine similarity ceiling for dedup (default 0.92)
  --max-retries N       Retries per comment slot on QC failure (default 3)

Generation options:
  --seed N              Random seed (default 42)
  --comment-period-days N  Simulated comment period length in days (default 60)
  --max-concurrent N    Max concurrent API requests (default 10)
  --no-async            Disable async parallelization

  --quiet               Suppress progress output
```

### `cli.py shuffle` — Shuffler

```
python cli.py shuffle [OPTIONS]

Required:
  --docket-id ID            Docket identifier — all paths derived automatically.
                            (or provide individual path arguments below)

Convention-based defaults (derived from --docket-id):
  --syncom-output PATH      Default: {docket_id}/synthetic_comments/synthetic.txt
  --translated-output PATH  Default: {docket_id}/shuffled_comments/synthetic_cms.csv
  --real-comments PATH      Default: {docket_id}/comments/{docket_id}.csv
  --combined-output PATH    Default: {docket_id}/shuffled_comments/combined.csv

Shuffle options:
  --key-output PATH         Path for the key CSV (default: <combined-stem>_key.csv).
  --skip-translation        Skip translation; use --translated-output as-is.
  --seed N                  Random seed for reproducible shuffling (default 42).

  --quiet                   Suppress progress output
```

### `campaign/planner.py` — Campaign Planner

```
python campaign/planner.py [OPTIONS]

Required:
  --docket-id ID            Docket identifier — all paths derived automatically.
                            (or provide individual path arguments below)

Convention-based defaults (derived from --docket-id):
  --scenario PATH_OR_TEXT   Default: {docket_id}/campaign/scenario_brief.txt
  --rule-text PATH_OR_TEXT  Default: {docket_id}/rule/rule.txt
  --output PATH             Default: {docket_id}/campaign/campaign_plan.json

API configuration:
  --api-base-url URL    Override SLOP_API_BASE_URL
  --api-key KEY         Override SLOP_API_KEY
  --chat-model MODEL    Override SLOP_CHAT_MODEL

  --quiet               Suppress progress output
```

## Architecture

### Component Overview

```
                         ┌─────────────────────────────────────────────────────┐
                         │                   SLOP Architecture                 │
                         └─────────────────────────────────────────────────────┘

 ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
 │  Downloader  │    │  Stylometry  │    │   Campaign   │    │   Shuffler   │
 │              │    │   Analyzer   │    │   Planner    │    │              │
 │ Download     │    │              │    │              │    │ Convert      │
 │ attachments  │    │ Extract      │    │ Scenario →   │    │ ♔-delimited  │
 │ from         │    │ voice skill  │    │ campaign_    │    │ → CMS CSV    │
 │ regulations  │    │ .md files    │    │ plan.json    │    │              │
 │ .gov         │    │              │    │              │    │              │
 └──────┬───────┘    └──────┬───────┘    └──────┬───────┘    └──────▲───────┘
        │                   │                   │                   │
        ▼                   ▼                   ▼                   │
 {docket}/          {docket}/           campaign_plan.json         │
 comment_           stylometry/                │                   │
 attachments/       *.md                       │                   │
                         │                     │                   │
                         └──────────┬──────────┘                   │
                                    ▼                               │
                          ┌─────────────────┐                      │
                          │     cli.py      │                      │
                          │                 │                      │
                          │  --docket-id    │                      │
                          │  --rule-text    │                      │
                          │  --vector       │                      │
                          │  --objective    │──────────────────────┘
                          │  --volume       │
                          │  --output       │
                          └────────┬────────┘
                                   │
                                   ▼
                          ┌─────────────────────────────────────────┐
                          │              syncom pipeline             │
                          │                                          │
                          │  World Model → Persona → Arguments →    │
                          │  Generator → Quality Control → Export   │
                          └─────────────────────────────────────────┘
```

### Key Components

1. **World Model** (`syncom/world_model.py`)
   - Extracts themes, stakeholders, and policy context from input documents
   - Builds a knowledge graph of the regulatory issue

2. **Persona Generator** (`syncom/persona.py`)
   - Creates realistic commenter profiles
   - Assigns location, occupation, personal hooks, sophistication level
   - Draws on stylometry voice skills for docket-specific style fidelity

3. **Argument Mapper** (`syncom/argument_mapper.py`)
   - Maps policy positions to specific stakeholder interests
   - Generates nuanced arguments that fit the persona
   - Uses campaign plan argument angles when `--campaign-plan` is provided

4. **Generator** (`syncom/generator.py`)
   - Writes the actual comment text using the configured LLM
   - Maintains consistency with persona, arguments, and voice skill

5. **Quality Control** (`syncom/quality_control.py`)
   - Validates output quality with parallel async checks
   - Checks for topical relevance, argument presence, and near-duplicate detection via embeddings

6. **Export** (`syncom/export.py`)
   - Formats output in ♔-delimited format with synthetic metadata fields

7. **Campaign Planner** (`campaign/planner.py`)
   - One-shot LLM call to convert a scenario brief into a structured JSON campaign plan
   - Plan controls argument angle distribution, archetype mix, and vector weighting

8. **Stylometry Analyzer** (`stylometry/stylometry_analyzer.py`)
   - Preprocesses real docket comments into per-voice-group skill files
   - Computes 20+ stylometric metrics; generates empirical AI-avoidance rules

## How It Works

### The Generation Process

1. **Load Voice Skills** - Read stylometry skill files for this docket (from `{docket_id}/stylometry/`)
2. **Analyze Context** - Read the rule text and extract policy themes
3. **Create Persona** - Generate a realistic commenter profile (archetype, location, occupation)
4. **Map Arguments** - Identify which policy arguments fit this persona (and campaign angle, if set)
5. **Generate Personal Hook** - Create a relatable story or experience
6. **Write Comment** - Compose the full comment using the LLM, guided by the voice skill
7. **Quality Check** - Run parallel async QC: relevance, argument presence, embedding dedup
8. **Export** - Format and save the result

### Sophistication Levels

**Low Sophistication**: Personal anecdotes, emotional appeals, simple language
```
"I'm worried about losing my insurance. My doctor uses these systems 
and I don't want anything to change that affects my care."
```

**Medium Sophistication**: Some policy knowledge, structured arguments
```
"Medicare Advantage plans use quality bonuses to fund digital health tools. 
Reducing these bonuses will limit investment in patient-facing technology 
and harm care coordination."
```

**High Sophistication**: Policy expertise, citations, technical details
```
"The proposed reduction in MA quality bonus payments under 42 CFR 422.166 
conflicts with interoperability requirements under the 21st Century Cures Act. 
Providers need stable reimbursement to absorb FHIR API implementation costs..."
```

## Output Format

Synthetic comments are saved in ♔-delimited format with added metadata:

```
Comment ID♔Document ID♔Submitter Name♔Organization♔...♔Comment♔synth_archetype♔synth_sophistication♔...
```

Key synthetic metadata fields:
- `synth_is_synthetic`: TRUE (always)
- `synth_archetype`: industry, individual_consumer, academia, advocacy_group, government
- `synth_sophistication`: low, medium, high
- `synth_emotional_register`: concerned, supportive, urgent, pragmatic
- `synth_persona_state`: US state (e.g., "Minnesota", "Ohio")
- `synth_persona_occupation`: Job title (e.g., "practice manager", "nurse")
- `synth_core_arguments`: Bullet points of main arguments
- `synth_qc_passed`: Quality control status

Use `shuffler/translate_to_cms_format.py` to convert this to standard CMS CSV format.

## Performance

- **Speed**: ~3-6 seconds per comment in async mode (10x parallel, GPT-4o)
- **Cost**: ~$0.05-0.15 per comment
- **Quality**: Passes human review in most cases
- **Diversity**: Varied personas, arguments, writing styles, and argument angles

Async mode (enabled by default) runs up to 10 comments concurrently. Use `--max-concurrent` to tune. See [`syncom/PERFORMANCE.md`](syncom/PERFORMANCE.md) for detailed benchmarks.

## Code Structure

```
slop/
├── cli.py                      # Main command-line interface
├── config.py                   # API configuration (reads .env / env vars)
├── shared_models.py            # Shared data models
├── requirements.txt            # Python dependencies
│
├── campaign/                   # Campaign Planner sub-application
│   ├── planner.py              # LLM-based scenario → campaign plan converter
│   ├── campaign_models.py      # Pydantic models for campaign plan schema
│   └── README.md
│
├── downloader/                 # Attachment Downloader sub-application
│   ├── download_attachments.py # Download attachments from regulations.gov
│   ├── text_converter.py       # Convert PDF/DOCX attachments to .txt
│   └── README_DOWNLOADER.md
│
├── shuffler/                   # Shuffler sub-application
│   ├── shuffler.py                 # Core shuffle logic (translate + interleave + key)
│   ├── translate_to_cms_format.py  # Convert ♔-delimited → CMS CSV
│   ├── verify_translation.py       # Verify translation output
│   └── TRANSLATION_README.md
│
├── stylometry/                 # Stylometry Analyzer sub-application
│   ├── stylometry_analyzer.py  # Analyze docket → generate voice skill files
│   ├── stylometry_loader.py    # Load voice skills at generation time
│   ├── stylometry_utils.py     # Shared metrics and utilities
│   └── STYLOMETRY_README.md
│
└── syncom/                     # Core Synthetic Comment Engine
    ├── pipeline.py             # Orchestrate generation (sync + async + campaign)
    ├── world_model.py          # Extract policy context from rule text
    ├── persona.py              # Generate commenter personas
    ├── argument_mapper.py      # Map arguments to personas
    ├── generator.py            # Write comments with LLM
    ├── quality_control.py      # Validate output (relevance, dedup, arguments)
    ├── export.py               # Format and save results
    └── ASYNC_IMPLEMENTATION.md
```

## Testing

```bash
# Test async functionality
python syncom/test_async.py

# Test attachment ingestion
python test_attachment_ingestion.py

# Test stylometry integration
python test_stylometry_integration.py
```

## Troubleshooting

### Common Issues

**"SLOP_API_KEY not set" / "API key not found"**
- Create a `.env` file with `SLOP_API_KEY=your_key_here` and `SLOP_EMBED_API_KEY=your_key_here`

**"No stylometry data found for docket"**
- Run `python stylometry/stylometry_analyzer.py {csv}` before generating comments

**"Generation too slow"**
- Async mode is enabled by default (10x concurrency). Use `--max-concurrent 15` for more aggression.
- Use a faster model by setting `SLOP_CHAT_MODEL=gpt-4o-mini`
- Reduce QC overhead with `--no-embedding-check` (disables dedup)

**"Output quality is poor"**
- Use GPT-4o instead of GPT-3.5 class models
- Run stylometry first for docket-specific voice skills
- Use a campaign plan for more focused argument angles

**API rate limit errors**
- Reduce concurrency: `--max-concurrent 5`
- Or fall back to sync: `--no-async`

## Theory & Background

### What is "SLOP"?

SLOP stands for "Synthetic Letter-writing Opposition Platform" — a tongue-in-cheek reference to the potential for AI-generated content to flood regulatory comment systems.

### The Signal vs. Slop Problem

Modern AI makes it trivially easy to generate sophisticated-looking public comments at scale. This creates challenges:

1. **Regulatory capture** - Well-resourced groups can flood dockets with AI-generated comments
2. **Drowning out authentic voices** - Real grassroots input gets lost in the noise
3. **Analysis paralysis** - Agencies must process thousands of low-signal comments

### Research Applications

This tool is designed to:
- **Study the problem** - Understand how AI-generated comments differ from authentic ones
- **Develop countermeasures** - Test detection methods and analysis pipelines
- **Educate regulators** - Demonstrate the scale and sophistication of the threat
- **Improve comment analysis** - Train better classifiers and summarization tools

### Ethical Considerations

**This tool should NOT be used to:**
- ❌ Submit fake comments to actual regulatory proceedings
- ❌ Deceive regulators or the public
- ❌ Astroturf or manipulate policy outcomes

**This tool IS intended for:**
- ✅ Academic research on computational persuasion
- ✅ Testing and improving comment analysis systems
- ✅ Training ML models to detect synthetic content
- ✅ Understanding regulatory capture dynamics

## Implementation Details

### Technology Stack

- **Language**: Python 3.8+
- **LLM API**: Any OpenAI-compatible chat/completions endpoint (GPT-4o recommended)
- **Embeddings API**: Any OpenAI-compatible embeddings endpoint (for QC deduplication)
- **Data Format**: CSV (CMS regulations.gov format) + ♔-delimited internal format
- **Key Dependencies**:
  - `openai` - LLM API access (sync + async)
  - `pandas` - Data manipulation
  - `numpy` - Stylometry statistics
  - `python-dotenv` - Environment variables
  - `pypdf`, `python-docx` - Attachment text extraction (optional)

## Contributing

Contributions are welcome! Areas for improvement:

- [ ] Automatic stylometry integration in generator (currently manual reference)
- [ ] Support for additional LLM providers
- [ ] Web interface for easier use
- [ ] Detection tools to identify synthetic comments
- [ ] Cross-docket stylometry comparison
- [ ] Multi-language support

## License

This project is provided for research and educational purposes. Please use responsibly and ethically.

## Citation

If you use this tool in research, please cite:

```
@software{slop2026,
  title={SLOP: Synthetic Letter-writing Opposition Platform},
  author={Kramer, Mark},
  year={2026},
  url={https://github.com/markkramerus/slop}
}
```

## Contact

- **GitHub**: https://github.com/markkramerus/slop
- **Issues**: https://github.com/markkramerus/slop/issues

## Acknowledgments

This tool was developed to study and address the challenge of AI-generated content in regulatory proceedings. Special thanks to the open-source community and regulatory researchers working on these critical issues.

---

**Remember**: With great power comes great responsibility. Use this tool ethically and in service of better democratic processes, not to undermine them.
