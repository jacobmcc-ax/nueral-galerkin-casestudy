# TDD Step 1: PDF to Markdown Conversion Verification

## Objective
Use Test-Driven Development to ensure complete and accurate extraction of the Neural Galerkin paper from PDF to Markdown format using Axiomatic-AI's document processing tools.

## TDD Cycle: Red → Green → Verify

### Phase 1: Pre-Conversion Tests (RED Phase)

Write tests that will initially FAIL before PDF conversion, defining our success criteria.

#### 1.1 Title Extraction Test
```python
# test_pdf_extraction.py
import pytest
from pathlib import Path

def test_title_extraction():
    """Test that paper title is correctly extracted from PDF"""
    # RED: This will fail initially - no markdown exists yet
    markdown_content = load_markdown_content("neural_galerkin_paper.md")

    expected_title = "Randomized Sparse Neural Galerkin Schemes for Solving Evolution Equations with Deep Networks"

    # Check title appears as main heading
    assert f"# {expected_title}" in markdown_content or f"#{expected_title}" in markdown_content

    # Check title formatting is clean (no PDF artifacts)
    title_line = extract_title_line(markdown_content)
    assert not contains_pdf_artifacts(title_line)  # No weird spacing, symbols
    assert title_line.count('#') >= 1  # Proper markdown heading
```

#### 1.2 Section Structure Test
```python
def test_section_structure():
    """Test that all major paper sections are extracted"""
    # RED: This will fail initially
    markdown_content = load_markdown_content("neural_galerkin_paper.md")
    sections = extract_sections(markdown_content)

    required_sections = [
        "Abstract",
        "Introduction",
        "Neural Galerkin Schemes",
        "Randomized Sparse Updates",
        "Experiments",
        "Results",
        "Conclusion",
        "References"
    ]

    for section in required_sections:
        assert section_exists(sections, section), f"Section '{section}' not found"
        assert proper_heading_level(sections, section), f"Section '{section}' has wrong heading level"
```

#### 1.3 Mathematical Content Test
```python
def test_mathematical_content_preservation():
    """Test that mathematical equations and formulas are preserved"""
    # RED: This will fail initially
    markdown_content = load_markdown_content("neural_galerkin_paper.md")

    # Check for mathematical notation preservation
    math_indicators = [
        "$",  # Inline math
        "$$", # Block math
        "\\begin{equation}",  # LaTeX equations
        "\\mathbf",  # Bold math symbols
        "\\partial",  # Partial derivatives
        "\\theta"  # Neural network parameters
    ]

    math_content_found = sum(1 for indicator in math_indicators if indicator in markdown_content)
    assert math_content_found >= 5, "Mathematical notation not properly preserved"

    # Check specific mathematical concepts from the paper
    mathematical_concepts = [
        "Galerkin projection",
        "sparse update",
        "randomization",
        "time evolution",
        "parameter subset"
    ]

    for concept in mathematical_concepts:
        assert concept.lower() in markdown_content.lower(), f"Mathematical concept '{concept}' missing"
```

#### 1.4 Algorithm/Pseudocode Test
```python
def test_algorithm_extraction():
    """Test that algorithms and pseudocode are properly formatted"""
    # RED: This will fail initially
    markdown_content = load_markdown_content("neural_galerkin_paper.md")

    # Look for algorithm blocks
    algorithm_indicators = [
        "Algorithm",
        "procedure",
        "function",
        "Input:",
        "Output:",
        "for each",
        "while",
        "return"
    ]

    # Check that at least one algorithm is present
    algorithm_content = extract_algorithm_blocks(markdown_content)
    assert len(algorithm_content) >= 1, "No algorithms found in markdown"

    # Check algorithm formatting
    for algorithm in algorithm_content:
        assert is_properly_formatted_code_block(algorithm), "Algorithm not in proper code block"
        assert contains_key_steps(algorithm), "Algorithm missing key procedural steps"
```

#### 1.5 Reference and Citation Test
```python
def test_references_and_citations():
    """Test that references and citations are preserved"""
    # RED: This will fail initially
    markdown_content = load_markdown_content("neural_galerkin_paper.md")

    # Check References section exists
    assert "References" in markdown_content or "Bibliography" in markdown_content

    # Check citation format is preserved
    reference_section = extract_references_section(markdown_content)
    references = parse_references(reference_section)

    # Should have reasonable number of references for academic paper
    assert len(references) >= 10, f"Too few references found: {len(references)}"

    # Check reference formatting
    for ref in references:
        assert has_author_info(ref), "Reference missing author information"
        assert has_publication_info(ref), "Reference missing publication details"

    # Check in-text citations exist
    citation_patterns = [r'\[(\d+)\]', r'\[\d+,\s*\d+\]', r'\[\d+-\d+\]']
    citations_found = count_citation_patterns(markdown_content, citation_patterns)
    assert citations_found >= 5, "Too few in-text citations found"
```

### Phase 2: Conversion Implementation (GREEN Phase)

#### 2.1 Execute PDF to Markdown Conversion
```bash
# Use Axiomatic-AI tools to convert PDF to Markdown
# This is the GREEN phase - implement minimal functionality to pass tests

# Example workflow:
# 1. Download PDF: https://arxiv.org/pdf/2310.04867
# 2. Use Axiomatic-AI PDF to Markdown conversion
# 3. Save as neural_galerkin_paper.md
# 4. Run tests to see which pass/fail
```

#### 2.2 Test Execution and Validation
```python
# Run the test suite
def execute_conversion_tests():
    """Execute all conversion tests and report results"""
    test_results = {
        'title_extraction': test_title_extraction(),
        'section_structure': test_section_structure(),
        'mathematical_content': test_mathematical_content_preservation(),
        'algorithm_extraction': test_algorithm_extraction(),
        'references_citations': test_references_and_citations()
    }

    return test_results

# Expected initial results (GREEN phase):
# - Some tests may pass immediately
# - Others may need markdown cleanup
# - Document any failing tests for refinement
```

### Phase 3: Refinement and Verification (REFACTOR)

#### 3.1 Quality Improvement Tests
```python
def test_markdown_quality():
    """Test overall markdown quality and readability"""
    markdown_content = load_markdown_content("neural_galerkin_paper.md")

    quality_checks = {
        'proper_heading_hierarchy': check_heading_hierarchy(markdown_content),
        'no_broken_links': verify_no_broken_links(markdown_content),
        'consistent_formatting': check_formatting_consistency(markdown_content),
        'readable_tables': verify_table_formatting(markdown_content),
        'image_references': check_image_references(markdown_content)
    }

    for check_name, result in quality_checks.items():
        assert result, f"Quality check failed: {check_name}"

def test_completeness_metrics():
    """Test that conversion captured reasonable percentage of content"""
    markdown_content = load_markdown_content("neural_galerkin_paper.md")

    # Rough metrics for completeness
    word_count = count_words(markdown_content)
    assert word_count >= 3000, f"Paper seems too short: {word_count} words"
    assert word_count <= 15000, f"Paper seems suspiciously long: {word_count} words"

    # Check content distribution
    content_metrics = analyze_content_distribution(markdown_content)
    assert content_metrics['text_percentage'] >= 70, "Too little text content"
    assert content_metrics['math_percentage'] >= 5, "Too little mathematical content"
```

#### 3.2 Technical Accuracy Validation
```python
def test_technical_term_preservation():
    """Test that technical terms are preserved correctly"""
    markdown_content = load_markdown_content("neural_galerkin_paper.md")

    critical_terms = [
        "Neural Galerkin schemes",
        "randomized sparse subsets",
        "evolution equations",
        "sequential-in-time training",
        "parameter updates",
        "time-dependent PDEs",
        "Dirac-Frenkel variational principle",
        "overfitting prevention"
    ]

    for term in critical_terms:
        assert term in markdown_content, f"Critical term missing: {term}"
        # Check term appears in proper context (not just isolated)
        assert appears_in_proper_context(markdown_content, term)
```

## Implementation Helper Functions

```python
# helper_functions.py
def load_markdown_content(filepath):
    """Load markdown file content"""
    return Path(filepath).read_text(encoding='utf-8')

def extract_title_line(content):
    """Extract the main title line from markdown"""
    lines = content.split('\n')
    for line in lines:
        if line.startswith('#') and not line.startswith('##'):
            return line.strip()
    return ""

def contains_pdf_artifacts(text):
    """Check for common PDF conversion artifacts"""
    artifacts = ['�', '...', '  ', '\x0c', 'ﬁ', 'ﬂ', '◦']
    return any(artifact in text for artifact in artifacts)

def extract_sections(content):
    """Extract section headings and their levels"""
    lines = content.split('\n')
    sections = []
    for line in lines:
        if line.strip().startswith('#'):
            level = len(line) - len(line.lstrip('#'))
            title = line.strip('#').strip()
            sections.append({'title': title, 'level': level})
    return sections

def section_exists(sections, section_name):
    """Check if a section exists in the extracted sections"""
    return any(section_name.lower() in section['title'].lower()
              for section in sections)

def proper_heading_level(sections, section_name):
    """Check if section has appropriate heading level"""
    for section in sections:
        if section_name.lower() in section['title'].lower():
            return 1 <= section['level'] <= 3
    return False

def extract_algorithm_blocks(content):
    """Extract code/algorithm blocks from markdown"""
    import re
    # Look for code blocks (``` or indented)
    code_blocks = re.findall(r'```[\s\S]*?```', content)
    indented_blocks = re.findall(r'(?:^    .*$\n?)+', content, re.MULTILINE)
    return code_blocks + indented_blocks

def is_properly_formatted_code_block(algorithm_text):
    """Check if algorithm is in proper markdown code block"""
    return algorithm_text.strip().startswith('```') or algorithm_text.startswith('    ')

def contains_key_steps(algorithm_text):
    """Check if algorithm contains procedural steps"""
    step_indicators = ['step', 'for', 'while', 'if', 'return', 'compute', 'update']
    return any(indicator in algorithm_text.lower() for indicator in step_indicators)
```

## Test Execution Workflow

### Step 1: Setup Test Environment
```bash
# Create test directory structure
mkdir -p tests/step1_pdf_conversion
cd tests/step1_pdf_conversion

# Install required testing dependencies
pip install pytest markdown beautifulsoup4 requests
```

### Step 2: Write Failing Tests
```bash
# Create test file with all RED phase tests
touch test_pdf_extraction.py
# Copy all test functions above into this file
```

### Step 3: Run Initial Tests (Should All Fail)
```bash
# These should all fail initially
pytest test_pdf_extraction.py -v

# Expected output:
# test_title_extraction FAILED
# test_section_structure FAILED
# test_mathematical_content_preservation FAILED
# test_algorithm_extraction FAILED
# test_references_and_citations FAILED
```

### Step 4: Execute PDF Conversion (GREEN Phase)
```bash
# Use Axiomatic-AI tools to convert PDF
# Save result as neural_galerkin_paper.md
```

### Step 5: Run Tests Again and Iterate
```bash
# Run tests to see progress
pytest test_pdf_extraction.py -v

# Fix any issues with markdown file
# Re-run tests until all pass
```

### Step 6: User Verification
- [ ] User reviews extracted markdown file
- [ ] User confirms title accuracy
- [ ] User validates section completeness
- [ ] User checks mathematical notation quality
- [ ] User verifies algorithm readability
- [ ] User confirms reference preservation

## Success Criteria

### All Tests Must Pass
- ✅ Title extracted correctly with proper formatting
- ✅ All major sections identified and structured
- ✅ Mathematical content preserved with proper notation
- ✅ Algorithms extracted in readable code blocks
- ✅ References and citations maintained
- ✅ Overall markdown quality meets standards
- ✅ Technical terms preserved accurately

### User Verification Complete
- ✅ User confirms extraction accuracy
- ✅ User validates technical content preservation
- ✅ User approves markdown quality for next development phase

## Next Steps

Once all Step 1 tests pass and user verification is complete:
1. Create `TDD-step2.md` for algorithm specification extraction
2. Use the validated markdown as input for algorithmic analysis
3. Begin Phase 2: Mathematical Foundation Tests from main TDD.md

This systematic approach ensures we have a high-quality, accurate markdown version of the paper before attempting to implement its algorithms.