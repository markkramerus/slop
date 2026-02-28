# Stylometry Analyzer - Quick Start Guide

## What is it?

A tool that analyzes real docket comments to extract writing style patterns and generate "voice skills" that make synthetic comments more realistic and docket-specific.

## Why use it?

Instead of using generic writing instructions, the stylometry analyzer creates docket-specific profiles based on actual commenters. Each profile includes:
- Statistical patterns (word count, sentence length)
- Structural patterns (bullets, headings, paragraphs)
- Vocabulary patterns (first-person usage, technical terms)
- AI-avoidance rules (what real commenters DON'T do)
- Real example excerpts

## Quick Start

### Step 1: Analyze a Docket

```bash
python stylometry/stylometry_analyzer.py CMS-2025-0050/comments/CMS-2025-0050.csv
```

**Output:**
```
INFO: Analyzed 980 comments
INFO: Identified 9 voice groups
INFO: Output directory: CMS-2025-0050/stylometry
```

**Generated files:**
- `CMS-2025-0050/stylometry/index.json` - Summary and mappings
- `CMS-2025-0050/stylometry/*.md` - Voice skill files (9 files in this case)

### Step 2: Examine the Skills

Each voice group gets its own markdown file:

```
CMS-2025-0050/
└── stylometry/
    ├── individual_consumer-low.md          (79 comments)
    ├── individual_consumer-medium.md       (491 comments)
    ├── individual_consumer-high.md         (58 comments)
    ├── industry-medium-org.md              (188 comments)
    ├── industry-high-org.md                (17 comments)
    ├── advocacy_group-medium-org.md        (112 comments)
    └── ...
```

### Step 3: Use with Generator (Optional)

**Option A: Manual Reference**

When generating comments, reference the appropriate skill file based on persona archetype and sophistication.

**Option B: Programmatic Integration**

```python
from syncom.stylometry_loader import load_voice_skill

# In your generation code:
voice_skill = load_voice_skill(
    docket_id="CMS-2025-0050-0031",
    archetype="individual_consumer",
    sophistication="low"
)

if voice_skill:
    # Use docket-specific skill in prompt
    style_instructions = voice_skill
else:
    # Fall back to generic instructions
    style_instructions = persona.style_instructions()
```

## What Makes a Voice Group?

Voice groups are based on **explicit CSV properties**:

1. **Archetype** (from CSV classification):
   - individual_consumer
   - industry
   - advocacy_group
   - academic
   - government

2. **Sophistication** (computed from fingerprints):
   - low: <200 words, simple vocabulary, no citations
   - medium: 200-400 words, moderate complexity
   - high: >400 words, citations, structured format

3. **Organization** (from CSV):
   - "-org" suffix if comment has organization name
   - No suffix for individual commenters

**Example Voice IDs:**
- `individual_consumer-low` → Short, simple personal comments
- `industry-high-org` → Long, formal organizational submissions
- `advocacy_group-medium-org` → Mid-length professional advocacy

## Example: CMS-2025-0050-0031 Results

From 980 comments, identified 9 distinct voices:

| Voice | Count | Avg Words | Structure | Citations |
|-------|-------|-----------|-----------|-----------|
| individual_consumer-medium | 491 | 267 | Some bullets | 0.1/comment |
| industry-medium-org | 188 | 328 | 54% bullets | 0.5/comment |
| advocacy_group-medium-org | 112 | 358 | 63% bullets | 0.7/comment |
| individual_consumer-low | 79 | 93 | 38% bullets | 0.0/comment |
| individual_consumer-high | 58 | 535 | 64% bullets | 2.2/comment |
| industry-high-org | 17 | 572 | 82% bullets | 1.1/comment |
| academic-medium-org | 13 | 362 | 46% bullets | 0.9/comment |

## What's in a Skill File?

Each `.md` file contains:

```markdown
# Voice Profile: Individual Consumer Low

## Statistical Profile
- Word count: 3-192 (mean: 93)
- Sentence length: 3-83 words (mean: 15)
- Paragraph count: 1-9 (mean: 3)

## Document Structure
- Uses bullet points: 38% of comments
- Average bullets: 1.5 per comment

## Vocabulary & Language
- First-person usage: 3.4% (I, me, my, we)
- Average word length: 4.7 characters
- Citations: 0.0 per comment

## Punctuation Patterns
- Em-dashes: 0.00 per 100 words
- Exclamation marks: 0.16 per 100 words

## AI-Avoidance Rules
- AVOID em-dashes — rarely used
- AVOID semicolons
- AVOID citing regulations
- Use first-person voice (I, me, my)

## Writing Instructions
1. Keep it simple: short sentences, everyday language
2. Be personal: start with your experience
3. Show emotion: frustration, worry, hope
4. Skip structure: no bullets or formal outline
5. Include imperfections: occasional typos
6. Length: Aim for 93 words (±51)

## Example Excerpts
> [Real samples from this voice group]
```

## Advanced Options

### Custom Output Directory

```bash
python stylometry_analyzer.py docket.csv --output-dir custom_dir/
```

### Adjust Minimum Group Size

```bash
# Only create skills for groups with ≥10 comments
python stylometry_analyzer.py docket.csv --min-group-size 10
```

### Check Available Skills

```python
from syncom.stylometry_loader import get_available_dockets

dockets = get_available_dockets()
print(f"Skills available for: {dockets}")
```

## Integration with Existing Pipeline

### Current Pipeline
1. Download docket → `python downloader/download_attachments.py {docket_id}/comments/{docket_id}.csv`
2. Generate comments → `cli.py` (uses generic `humanizer-skill.md`)

### Enhanced Pipeline
1. Download docket and convert to text → `python downloader/download_attachments.py {docket_id}/comments/{docket_id}.csv --convert-text`
2. **Analyze stylometry** → `python stylometry/stylometry_analyzer.py {docket_id}/comments/{docket_id}.csv`
3. Generate comments → `cli.py` (references `{docket_id}/stylometry/`)

## Tips

### Realistic Comment Generation
- **Use docket-specific skills** instead of generic instructions
- **Match persona sophistication** to appropriate skill level
- **Include example excerpts** in prompts for style priming

### Quality Analysis
- **Check sample sizes**: Groups with >50 samples are more reliable
- **Review AI markers**: High AI marker % may indicate contaminated training data
- **Compare across archetypes**: Different dockets have different patterns

### Troubleshooting
- **"No voice groups created"**: Check min-group-size or CSV format
- **Skills seem too similar**: May need to refine sophistication heuristics
- **Missing archetypes**: Some dockets may lack certain commenter types

## Files Created

```
{docket-id}/
└── stylometry/
    ├── index.json                          # Summary metadata
    ├── individual_consumer-low.md          # Voice skill
    ├── individual_consumer-medium.md       # Voice skill
    ├── individual_consumer-high.md         # Voice skill
    ├── industry-medium-org.md              # Voice skill
    ├── industry-high-org.md                # Voice skill
    ├── advocacy_group-medium-org.md        # Voice skill
    └── ...
```

## Next Steps

1. **Run analyzer** on your docket CSV
2. **Review generated skills** to understand commenter patterns
3. **Test integration** with comment generator
4. **Compare quality** of generic vs. docket-specific synthetic comments

## More Information

- Full documentation: `STYLOMETRY_README.md`
- Integration examples: `syncom/stylometry_loader.py`
- Analyzer source: `stylometry_analyzer.py`

## Key Insight

Real commenters in healthcare dockets write differently than those in environmental dockets. The stylometry analyzer captures these domain-specific patterns automatically, making synthetic comments more realistic and harder to detect.
