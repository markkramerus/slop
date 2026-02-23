# SLOP - Synthetic Letter-writing Opposition Platform

## Quick Start

### Generate synthetic public comments in 3 steps:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set up your OpenAI API key
# Create a .env file with: OPENAI_API_KEY=your_key_here

# 3. Generate synthetic comments
python cli.py generate --input CMS-2025-0050-0031.csv --output synthetic_comments.txt --count 10
```

That's it! You'll have 10 synthetic public comments that look and feel authentic.

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
- OpenAI API key (GPT-4 recommended)

### Install

```bash
# Clone the repository
git clone https://github.com/markkramerus/slop.git
cd slop

# Install dependencies
pip install -r requirements.txt

# Set up your API key
echo "OPENAI_API_KEY=your_key_here" > .env
```

## Basic Usage

### Generate Comments

```bash
# Generate 5 comments from a CSV input
python cli.py generate --input input.csv --output results.txt --count 5

# Specify a particular docket
python cli.py generate --input CMS-2025-0050-0031.csv --count 10

# Use a specific seed document (row number)
python cli.py generate --input input.csv --seed 3 --count 5
```

### Translate Back to CMS Format

After generating synthetic comments, convert them back to standard CMS CSV format:

```bash
# Translate to CMS format
python translate_to_cms_format.py synthetic_comments.txt output_cms.csv

# Verify the translation
python verify_translation.py
```

## Command-Line Options

```
python cli.py generate [OPTIONS]

Options:
  --input TEXT          Input CSV file (CMS format)
  --output TEXT         Output file for synthetic comments
  --seed INTEGER        Row number to use as seed document
  --count INTEGER       Number of comments to generate (default: 1)
  --model TEXT          OpenAI model to use (default: gpt-4)
  --temperature FLOAT   Temperature for generation (default: 0.8)
  --help               Show this message and exit
```

## Configuration

Edit `slop/config.py` to customize:

```python
# Model settings
MODEL = "gpt-4-turbo-preview"  # or "gpt-4", "gpt-3.5-turbo"
TEMPERATURE = 0.8              # Higher = more creative
MAX_TOKENS = 2000             # Maximum comment length

# Persona settings
ARCHETYPES = [                # Types of commenters
    "industry",
    "individual_consumer", 
    "academia",
    "advocacy"
]

EMOTIONAL_REGISTERS = [       # Emotional tones
    "concerned",
    "supportive",
    "urgent",
    "pragmatic"
]
```

## Examples

### Example 1: Generate Industry Perspectives

```bash
# Generate 5 healthcare industry comments
python cli.py generate \
  --input CMS-2025-0050-0031.csv \
  --count 5 \
  --output industry_comments.txt
```

### Example 2: Stress-Test Analysis Pipeline

```bash
# Generate 100 diverse comments for testing
python cli.py generate \
  --input regulatory_notice.csv \
  --count 100 \
  --output test_corpus.txt
```

### Example 3: Research Different Voices

```bash
# Generate comments with varying sophistication levels
# The system automatically varies:
# - Sophistication (low/medium/high)
# - Emotional register (concerned/supportive/urgent)
# - Archetype (industry/consumer/academic/advocacy)
```

## Output Format

Synthetic comments are saved in ♔-delimited format (same as input but with added metadata):

```
Comment ID♔Document ID♔Submitter Name♔Organization♔...♔Comment♔synth_archetype♔synth_sophistication♔...
```

Key synthetic metadata fields:
- `synth_is_synthetic`: TRUE (always)
- `synth_archetype`: industry, individual_consumer, academia, advocacy
- `synth_sophistication`: low, medium, high
- `synth_emotional_register`: concerned, supportive, urgent, pragmatic
- `synth_persona_state`: US state (e.g., "Minnesota", "Ohio")
- `synth_persona_occupation`: Job title (e.g., "practice manager", "nurse")
- `synth_core_arguments`: Bullet points of main arguments
- `synth_qc_passed`: Quality control status

## Architecture

### Component Overview

```
Input CSV → Pipeline → Personas → Arguments → Generation → QC → Output
            ↓
         World Model
```

### Key Components

1. **World Model** (`slop/world_model.py`)
   - Extracts themes, stakeholders, and policy context from input documents
   - Builds a knowledge graph of the regulatory issue

2. **Persona Generator** (`slop/persona.py`)
   - Creates realistic commenter profiles
   - Assigns location, occupation, personal hooks, sophistication level

3. **Argument Mapper** (`slop/argument_mapper.py`)
   - Maps policy positions to specific stakeholder interests
   - Generates nuanced arguments that fit the persona

4. **Generator** (`slop/generator.py`)
   - Uses GPT-4 to write the actual comment text
   - Maintains consistency with persona and arguments

5. **Quality Control** (`slop/quality_control.py`)
   - Validates output quality
   - Checks for coherence, authenticity, and policy relevance

6. **Export** (`slop/export.py`)
   - Formats output for CMS regulations.gov format
   - Handles metadata and field mapping

## How It Works

### The Generation Process

1. **Analyze Context** - Read the seed document and extract policy themes
2. **Create Persona** - Generate a realistic commenter profile (e.g., "practice manager in Minnesota")
3. **Map Arguments** - Identify which policy arguments fit this persona
4. **Generate Personal Hook** - Create a relatable story or experience
5. **Write Comment** - Use GPT-4 to compose the full comment
6. **Quality Check** - Validate authenticity and coherence
7. **Export** - Format and save the result

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

## Performance

- **Speed**: ~30-60 seconds per comment (using GPT-4)
- **Cost**: ~$0.05-0.15 per comment (GPT-4 pricing)
- **Quality**: Passes human review in most cases
- **Diversity**: Generates varied personas, arguments, and writing styles

See `PERFORMANCE.md` for detailed benchmarks.

## Advanced Features

### Async Generation

For bulk generation, the pipeline supports asynchronous processing:

```python
from slop.pipeline import Pipeline
import asyncio

async def generate_bulk():
    pipeline = Pipeline(input_file="input.csv")
    results = await pipeline.generate_async(count=100)
    return results

# Run async
results = asyncio.run(generate_bulk())
```

See `ASYNC_IMPLEMENTATION.md` for details.

### Custom Archetypes

Add your own commenter archetypes in `slop/config.py`:

```python
ARCHETYPE_PROMPTS = {
    "industry": "healthcare industry representative",
    "consumer": "individual patient or caregiver",
    "custom_advocacy": "environmental advocacy group focused on health equity"
}
```

### Translation Tools

Convert synthetic comments back to standard CMS CSV format:

- `translate_to_cms_format.py` - Main translation script
- `verify_translation.py` - Verification tool
- See `TRANSLATION_README.md` for details

## Theory & Background

### What is "SLOP"?

SLOP stands for "Synthetic Letter-writing Opposition Platform" - a tongue-in-cheek reference to the potential for AI-generated content to flood regulatory comment systems.

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
- **LLM API**: OpenAI GPT-4 (or GPT-3.5-turbo)
- **Data Format**: CSV (CMS regulations.gov format)
- **Dependencies**: 
  - `openai` - GPT API access
  - `pandas` - Data manipulation
  - `python-dotenv` - Environment variables
  - Standard library: `csv`, `json`, `asyncio`, etc.

### Code Structure

```
slop/
├── config.py          # Configuration settings
├── world_model.py     # Extract policy context
├── persona.py         # Generate commenter personas
├── argument_mapper.py # Map arguments to personas
├── generator.py       # Write comments with GPT-4
├── quality_control.py # Validate output quality
├── export.py          # Format and save results
└── pipeline.py        # Orchestrate the process

cli.py                 # Command-line interface
translate_to_cms_format.py  # Convert back to CMS format
verify_translation.py  # Verify translations
```

### Data Flow

```
1. Input CSV (CMS format)
   ↓
2. World Model extracts themes, stakeholders, policy context
   ↓
3. Persona Generator creates commenter profile
   ↓
4. Argument Mapper selects relevant arguments
   ↓
5. Generator composes comment with GPT-4
   ↓
6. Quality Control validates output
   ↓
7. Export formats as ♔-delimited file
```

## Testing

Run the test suite:

```bash
# Test async functionality
python test_async.py

# Test translation
python translate_to_cms_format.py synthetic_comments_3.txt test_output.csv
python verify_translation.py
```

## Troubleshooting

### Common Issues

**"OpenAI API key not found"**
- Create a `.env` file with `OPENAI_API_KEY=your_key_here`

**"Input file not found"**
- Verify the CSV file path is correct
- Check that the file uses standard CMS format

**"Generation too slow"**
- Use GPT-3.5-turbo instead of GPT-4 (faster but lower quality)
- Reduce `MAX_TOKENS` in config.py
- Use async generation for bulk processing

**"Output quality is poor"**
- Increase temperature for more creative output
- Use GPT-4 instead of GPT-3.5-turbo
- Adjust sophistication settings in config.py

## Contributing

Contributions are welcome! Areas for improvement:

- [ ] Add more archetype variations
- [ ] Implement better quality control metrics
- [ ] Support for other LLM providers (Anthropic, local models)
- [ ] Web interface for easier use
- [ ] Detection tools to identify synthetic comments
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
