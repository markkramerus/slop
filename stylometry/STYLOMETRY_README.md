# Stylometry Analyzer

A preprocessing tool that analyzes writing styles in real docket comments and generates reusable "voice skill" markdown files for generating more realistic synthetic comments.

## Overview

The stylometry analyzer separates stylistic and statistical analysis from the synthetic comment generation pipeline. Instead of using generic writing instructions, it extracts actual writing patterns from real commenters and encodes them as skills that the comment generator can reference.

## Key Concepts

### Voice Groups

Comments are grouped by **explicit CSV properties** (not clustering):
- **Archetype**: individual_consumer, advocacy_group, industry, academic, government
- **Sophistication level**: low, medium, high (computed from word count, citations, structure)
- **Organization presence**: has organization name vs. individual

Examples:
- `individual_consumer-low`: Short, simple, personal comments
- `industry-high-org`: Long, formal, structured organizational comments
- `advocacy_group-medium-org`: Mid-length, professional advocacy comments

### Skill Files

Each voice group gets a `.md` skill file containing:

1. **Statistical Profile**: Word count ranges, sentence lengths, paragraph counts
2. **Document Structure**: Bullet point usage, heading frequency, formatting
3. **Vocabulary & Language**: First-person usage, technical terminology, word length
4. **Punctuation Patterns**: Em-dash, semicolon, exclamation frequency
5. **AI Writing Markers**: Detection of AI vocabulary and phrases
6. **AI-Avoidance Rules**: Custom rules based on what this voice DOESN'T use
7. **Writing Instructions**: How to write in this voice
8. **Example Excerpts**: Real samples from this voice group

## Usage

### Basic Analysis

Analyze a docket and generate skills:

```bash
python stylometry/stylometry_analyzer.py CMS-2025-0050/comments/CMS-2025-0050.csv
```

This creates:
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

### Advanced Options

```bash
# Custom output directory
python stylometry/stylometry_analyzer.py docket.csv --output-dir custom_dir/

# Custom minimum group size (default: 5)
python stylometry/stylometry_analyzer.py docket.csv --min-group-size 10

# Specify attachments directory
python stylometry/stylometry_analyzer.py docket.csv --attachments-dir custom/attachments/
```

## Output Structure

### index.json

The index file provides a summary and archetype mapping:

```json
{
  "docket_id": "CMS-2025-0050-0031",
  "analyzed_at": "2026-02-25T15:34:15",
  "total_comments": 980,
  "voice_groups": [
    {
      "voice_id": "individual_consumer-low",
      "filename": "individual_consumer-low.md",
      "archetype": "individual_consumer",
      "sophistication": "low",
      "sample_size": 79
    }
  ],
  "archetype_mapping": {
    "individual_consumer": [
      "individual_consumer-low.md",
      "individual_consumer-medium.md",
      "individual_consumer-high.md"
    ],
    "industry": [...]
  }
}
```

### Skill Markdown Files

Each skill file contains actionable writing instructions. Example structure:

```markdown
---
name: individual_consumer-low
docket: CMS-2025-0050-0031
archetype: individual_consumer
sophistication: low
sample_size: 79
---

# Voice Profile: Individual Consumer Low

## Statistical Profile
- Word count: 3-192 (mean: 93, std: 51)
- Sentence length: 3-83 (mean: 15, std: 10) words
...

## AI-Avoidance Rules
- AVOID em-dashes (—) - rarely used in this voice
- AVOID semicolons - not typical for this voice
...

## Writing Instructions
1. Keep it simple: Use short sentences and everyday language
2. Be personal: Start with your own experience
...

## Example Excerpts
> [Real excerpts from this voice group]
```

## Integration with Comment Generator

### Option 1: Manual Reference (Current)

When generating comments, manually select and reference appropriate skill files based on the persona's archetype and sophistication level.

### Option 2: Automatic Integration (Future Enhancement)

Modify `syncom/generator.py` to:
1. Check if stylometry skills exist for the docket
2. Select appropriate skill based on persona archetype/sophistication
3. Include skill content in generation prompt instead of generic `humanizer-skill.md`

Example integration code:

```python
def load_voice_skill(docket_id: str, archetype: str, sophistication: str) -> str:
    """Load the appropriate voice skill for a persona."""
    skill_dir = Path("stylometry") / docket_id
    index_path = skill_dir / "index.json"
    
    if not index_path.exists():
        return ""  # Fall back to generic humanizer-skill.md
    
    with open(index_path) as f:
        index = json.load(f)
    
    # Find matching skill file
    for skill in index["voice_groups"]:
        if (skill["archetype"] == archetype and 
            skill["sophistication"] == sophistication):
            skill_path = skill_dir / skill["filename"]
            return skill_path.read_text()
    
    return ""
```

## Workflow

### Standard Pipeline (Before Stylometry)

1. Download docket → `{docket-id}/comment_attachments/`
2. Generate synthetic comments → Uses generic `humanizer-skill.md`

### Enhanced Pipeline (With Stylometry)

1. Download docket and convert to text → `python downloader/download_attachments.py {csv} --convert-text`
2. **Analyze stylometry** → `python stylometry/stylometry_analyzer.py {csv}` → `{docket-id}/stylometry/*.md`
3. Generate synthetic comments → Uses docket-specific voice skills

## Analysis Methodology

### Sophistication Classification

Comments are classified as low/medium/high based on:

- **High**: >400 words + (>2 citations OR >30% bullets)
- **Low**: <200 words + <5 char avg word length + 0 citations
- **Medium**: Everything else

### Metrics Captured

The analyzer computes 20+ stylometric metrics:

**Base metrics** (from `fingerprint()`):
- Word count, sentence count, mean sentence length
- Mean word length, first-person ratio
- Bullet ratio, citation count

**Enhanced metrics** (new):
- Punctuation frequencies (em-dash, ellipsis, exclamation, etc.)
- Document structure (paragraphs, bullets, headings)
- AI vocabulary detection
- Error patterns (typos, capitalization)

### Voice Group Size Filtering

Only voice groups with ≥5 comments generate skill files (configurable with `--min-group-size`). This ensures statistical validity.

## Examples from CMS-2025-0050-0031

From analyzing 980 comments, we identified 9 voice groups:

| Voice Group | Sample Size | Description |
|-------------|-------------|-------------|
| individual_consumer-medium | 491 | Most common: mid-length personal comments |
| industry-medium-org | 188 | Professional organizational comments |
| advocacy_group-medium-org | 112 | Advocacy group standard voice |
| individual_consumer-low | 79 | Short, simple personal comments |
| individual_consumer-high | 58 | Long, detailed individual comments |
| industry-high-org | 17 | Highly formal corporate submissions |
| academic-medium-org | 13 | Academic/research comments |
| industry-low-org | 7 | Brief organizational notes |
| advocacy_group-high-org | 6 | Formal advocacy submissions |

## Advantages Over Generic Instructions

### Before (Generic)
- All personas in an archetype use the same generic `humanizer-skill.md`
- No docket-specific stylistic patterns
- AI-avoidance rules are universal, not tailored

### After (Docket-Specific)
- Each voice uses patterns from real commenters in that docket
- Captures domain-specific vocabulary and structure
- AI-avoidance rules based on what real commenters DON'T do
- Statistical targets (word count, sentence length) are empirical

## Dependencies

The analyzer reuses existing dependencies from `syncom/ingestion.py`:
- pandas
- numpy
- pypdf (optional, for attachment extraction)
- python-docx (optional, for attachment extraction)

No additional packages required.

## Limitations

1. **Minimum data requirement**: Needs at least 5 comments per voice group
2. **Static analysis**: Does not update automatically when new comments are added
3. **Sophistication heuristics**: Classification is rule-based, may not capture all nuances
4. **No semantic analysis**: Focuses on style, not content/arguments

## Future Enhancements

1. **Automatic integration with generator.py**: Select skills automatically based on persona
2. **Domain-specific vocabulary extraction**: Identify technical terms unique to each docket
3. **Temporal analysis**: Track style changes over time within a docket
4. **Cross-docket comparison**: Compare writing styles across different regulatory domains
5. **Interactive skill refinement**: Allow manual adjustment of generated skills

## Comparison to Ingestion.py

| Aspect | ingestion.py | stylometry_analyzer.py |
|--------|-------------|------------------------|
| **Purpose** | Build population model for generation | Extract writing style profiles |
| **Output** | `PopulationModel` (Python object) | Markdown skill files |
| **Usage** | Runtime (during generation) | Preprocessing (before generation) |
| **Focus** | Demographics, archetypes, timing | Writing style, vocabulary, structure |
| **Integration** | Required for pipeline | Optional enhancement |

Both tools are complementary:
- **ingestion.py**: Provides demographic distributions and archetype profiles
- **stylometry_analyzer.py**: Provides writing style instructions for each archetype

## Troubleshooting

### "Could not find comment column"
Ensure your CSV has a column like "Comment", "Abstract", or "Comment Text".

### "No voice groups created"
- Check `--min-group-size` setting (default: 5)
- Verify CSV has enough valid comments (>10 characters each)
- Check that archetypes are being classified correctly

### Skills seem generic
- Increase `--min-group-size` for more stable statistics
- Consider merging similar sophistication levels if sample sizes are small

## License

Same as parent project.
