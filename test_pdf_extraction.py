#!/usr/bin/env python3
"""
Test-Driven Development for PDF to Markdown Extraction
RED Phase: All tests should FAIL initially before PDF conversion

This test suite validates that the Neural Galerkin paper PDF is accurately
converted to Markdown format, preserving all critical content and structure.
"""

import pytest
import re
from pathlib import Path
from typing import List, Dict, Any


# Helper Functions
def load_markdown_content(filepath: str) -> str:
    """Load markdown file content safely"""
    try:
        return Path(filepath).read_text(encoding='utf-8')
    except FileNotFoundError:
        pytest.fail(f"Markdown file not found: {filepath}")
    except Exception as e:
        pytest.fail(f"Error loading markdown file: {e}")


def extract_title_line(content: str) -> str:
    """Extract the main title line from markdown"""
    lines = content.split('\n')
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('#') and not stripped.startswith('##'):
            return stripped
    return ""


def contains_pdf_artifacts(text: str) -> bool:
    """Check for common PDF conversion artifacts"""
    artifacts = ['�', '…', '  ', '\x0c', 'ﬁ', 'ﬂ', '◦', '□', '■']
    return any(artifact in text for artifact in artifacts)


def extract_sections(content: str) -> List[Dict[str, Any]]:
    """Extract section headings and their levels"""
    lines = content.split('\n')
    sections = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('#'):
            level = len(line) - len(line.lstrip('#'))
            title = stripped.lstrip('#').strip()
            sections.append({'title': title, 'level': level})
    return sections


def section_exists(sections: List[Dict], section_name: str) -> bool:
    """Check if a section exists in the extracted sections"""
    return any(section_name.lower() in section['title'].lower()
              for section in sections)


def proper_heading_level(sections: List[Dict], section_name: str) -> bool:
    """Check if section has appropriate heading level"""
    for section in sections:
        if section_name.lower() in section['title'].lower():
            return 1 <= section['level'] <= 3
    return False


def extract_algorithm_blocks(content: str) -> List[str]:
    """Extract code/algorithm blocks from markdown"""
    # Look for code blocks (``` or indented)
    code_blocks = re.findall(r'```[\s\S]*?```', content)
    indented_blocks = re.findall(r'(?:^    .*$\n?)+', content, re.MULTILINE)
    return code_blocks + indented_blocks


def is_properly_formatted_code_block(algorithm_text: str) -> bool:
    """Check if algorithm is in proper markdown code block"""
    return (algorithm_text.strip().startswith('```') or
            algorithm_text.startswith('    ') or
            '```' in algorithm_text)


def contains_key_steps(algorithm_text: str) -> bool:
    """Check if algorithm contains procedural steps"""
    step_indicators = ['step', 'for', 'while', 'if', 'return', 'compute', 'update',
                      'algorithm', 'procedure', 'function', 'input:', 'output:']
    return any(indicator in algorithm_text.lower() for indicator in step_indicators)


def extract_references_section(content: str) -> str:
    """Extract the references section from markdown"""
    lines = content.split('\n')
    in_references = False
    references_content = []

    for line in lines:
        if ('references' in line.lower() or 'bibliography' in line.lower()) and line.startswith('#'):
            in_references = True
            continue
        if in_references:
            if line.startswith('#') and not line.startswith('##'):
                break  # Next major section
            references_content.append(line)

    return '\n'.join(references_content)


def parse_references(ref_section: str) -> List[str]:
    """Parse individual references from references section"""
    import re

    # First try to split by numbered references [1], [2], etc.
    pattern = r'\[(\d+)\]'
    parts = re.split(pattern, ref_section)

    references = []
    # Skip the first part (before first reference)
    for i in range(1, len(parts), 2):
        if i + 1 < len(parts):
            ref_num = parts[i]
            ref_content = parts[i + 1].strip()
            if ref_content:
                full_ref = f'[{ref_num}] {ref_content}'
                references.append(full_ref)

    # If no numbered references found, fall back to line-based parsing
    if not references:
        lines = ref_section.split('\n')
        current_ref = []
        for line in lines:
            line = line.strip()
            if not line:
                if current_ref:
                    references.append('\n'.join(current_ref))
                    current_ref = []
            else:
                current_ref.append(line)
        if current_ref:
            references.append('\n'.join(current_ref))

    return [ref for ref in references if len(ref.strip()) > 10]


def has_author_info(reference: str) -> bool:
    """Check if reference contains author information"""
    # Look for common author patterns
    author_patterns = [
        r'[A-Z][a-z]+,?\s+[A-Z]\.?',  # Last, F.
        r'[A-Z]\.\s+[A-Z][a-z]+',     # F. Last
        r'et al\.?',                   # et al.
    ]
    return any(re.search(pattern, reference) for pattern in author_patterns)


def has_publication_info(reference: str) -> bool:
    """Check if reference contains publication information"""
    pub_indicators = ['journal', 'conference', 'proceedings', 'arxiv', 'doi:', 'isbn:',
                     'volume', 'pages', 'pp\.', '\d{4}', 'publisher']
    return any(re.search(indicator, reference, re.IGNORECASE) for indicator in pub_indicators)


def count_citation_patterns(content: str, patterns: List[str]) -> int:
    """Count in-text citations using regex patterns"""
    total_count = 0
    for pattern in patterns:
        matches = re.findall(pattern, content)
        total_count += len(matches)
    return total_count


def check_heading_hierarchy(content: str) -> bool:
    """Check if heading hierarchy is logical"""
    sections = extract_sections(content)
    if not sections:
        return False

    # Check that we have a main title (level 1)
    has_main_title = any(section['level'] == 1 for section in sections)

    # Check that heading levels don't skip (no level 1 directly to level 3)
    levels = [section['level'] for section in sections]
    for i in range(1, len(levels)):
        if levels[i] - levels[i-1] > 1:
            return False

    return has_main_title


def verify_no_broken_links(content: str) -> bool:
    """Basic check for obviously broken links"""
    # Look for broken link patterns
    broken_patterns = [r'\[.*\]\(\)', r'\[.*\]\(#\)', r'!\[.*\]\(\)']
    for pattern in broken_patterns:
        if re.search(pattern, content):
            return False
    return True


def check_formatting_consistency(content: str) -> bool:
    """Check for consistent markdown formatting"""
    lines = content.split('\n')

    # Check for consistent heading styles
    heading_styles = set()
    for line in lines:
        if line.strip().startswith('#'):
            # Count spaces after #
            hash_count = len(line) - len(line.lstrip('#'))
            space_after = line[hash_count:hash_count+1] == ' ' if len(line) > hash_count else False
            heading_styles.add(space_after)

    # Should have consistent heading style
    return len(heading_styles) <= 1


def verify_table_formatting(content: str) -> bool:
    """Check if tables are properly formatted"""
    # Look for table patterns
    table_lines = [line for line in content.split('\n') if '|' in line]
    if not table_lines:
        return True  # No tables is fine

    # Check for table headers (should have separator line)
    for i, line in enumerate(table_lines[:-1]):
        if i+1 < len(table_lines):
            next_line = table_lines[i+1]
            if ('---' in next_line or '===' in next_line or
                ':--' in next_line or '--:' in next_line or
                ':-:' in next_line):
                return True

    return len(table_lines) == 0  # If no proper tables, that's okay


def check_image_references(content: str) -> bool:
    """Check if image references are properly formatted"""
    # Look for image patterns
    image_patterns = [r'!\[.*\]\(.*\)', r'<img.*>']
    images_found = sum(len(re.findall(pattern, content)) for pattern in image_patterns)

    # Check that images have alt text if they exist
    if images_found > 0:
        alt_text_pattern = r'!\[.+\]'
        alt_texts = len(re.findall(alt_text_pattern, content))
        return alt_texts >= images_found * 0.8  # 80% should have alt text

    return True


def count_words(content: str) -> int:
    """Count words in content"""
    # Remove markdown formatting for word count
    text = re.sub(r'[#*`_\[\]()]', ' ', content)
    words = text.split()
    return len([word for word in words if len(word) > 1])


def analyze_content_distribution(content: str) -> Dict[str, float]:
    """Analyze distribution of content types"""
    total_chars = len(content)

    # Count mathematical content
    math_chars = len(re.findall(r'\$.*?\$', content, re.DOTALL))
    math_chars += len(re.findall(r'\$\$.*?\$\$', content, re.DOTALL))
    math_chars += len(re.findall(r'\\[a-zA-Z]+', content))

    # Count code content
    code_chars = len(re.findall(r'```.*?```', content, re.DOTALL))
    code_chars += len(re.findall(r'`.*?`', content))

    # Count regular text (approximate)
    text_chars = total_chars - math_chars - code_chars

    if total_chars == 0:
        return {'text_percentage': 0, 'math_percentage': 0, 'code_percentage': 0}

    return {
        'text_percentage': (text_chars / total_chars) * 100,
        'math_percentage': (math_chars / total_chars) * 100,
        'code_percentage': (code_chars / total_chars) * 100
    }


def appears_in_proper_context(content: str, term: str) -> bool:
    """Check if term appears in proper context (not isolated)"""
    # Find sentences containing the term
    sentences = re.split(r'[.!?]+', content)
    term_sentences = [s for s in sentences if term.lower() in s.lower()]

    # Term should appear in sentences with reasonable length
    return any(len(s.split()) >= 5 for s in term_sentences)


# Test Suite (RED Phase - All should FAIL initially)
class TestPDFExtraction:
    """Test suite for PDF to Markdown extraction validation"""

    EXPECTED_MARKDOWN_FILE = "/Users/jacobmccarran_ax/Downloads/nueral-galerkin-casestudy/nueral-galerkin.md"

    @pytest.fixture
    def markdown_content(self):
        """Load the converted markdown content"""
        return load_markdown_content(self.EXPECTED_MARKDOWN_FILE)

    def test_title_extraction(self, markdown_content):
        """Test that paper title is correctly extracted from PDF"""
        print("\n" + "="*60)
        print("TITLE EXTRACTION TEST - DETAILED ANALYSIS")
        print("="*60)

        expected_title = "Randomized Sparse Neural Galerkin Schemes for Solving Evolution Equations with Deep Networks"
        print(f"Expected title: '{expected_title}'")

        # Check title appears as main heading
        title_variations = [
            f"# {expected_title}",
            f"#{expected_title}",
            expected_title  # Title might appear without # if formatted differently
        ]

        print(f"\nSearching for {len(title_variations)} title variations:")
        found_variations = []
        for i, variation in enumerate(title_variations, 1):
            if variation in markdown_content:
                found_variations.append(variation)
                print(f"  ✅ Variation {i}: Found - '{variation[:50]}{'...' if len(variation) > 50 else ''}'")
            else:
                print(f"  ❌ Variation {i}: Not found - '{variation[:50]}{'...' if len(variation) > 50 else ''}'")

        title_found = len(found_variations) > 0
        print(f"\n📊 TITLE MATCH RESULT: {len(found_variations)}/{len(title_variations)} variations found")
        assert title_found, f"Expected title '{expected_title}' not found in any of {len(title_variations)} variations"

        # Check title formatting is clean (no PDF artifacts)
        title_line = extract_title_line(markdown_content)
        print(f"\n🔍 EXTRACTED TITLE LINE: '{title_line}'")

        if not title_line:
            print("  ❌ ERROR: No H1 heading line found in document")
            assert False, "No title line found"
        else:
            print(f"  ✅ H1 heading found: '{title_line}'")

        # Check for PDF artifacts
        artifacts_found = []
        pdf_artifacts = ['�', '…', '  ', '\x0c', 'ﬁ', 'ﬂ', '◦', '□', '■']
        for artifact in pdf_artifacts:
            if artifact in title_line:
                artifacts_found.append(artifact)

        print(f"\n🧹 PDF ARTIFACTS CHECK:")
        if artifacts_found:
            print(f"  ❌ Found {len(artifacts_found)} artifacts: {artifacts_found}")
            assert False, f"Title contains PDF artifacts: {title_line}"
        else:
            print(f"  ✅ No PDF artifacts detected in title")

        # Check heading format
        hash_count = title_line.count('#')
        print(f"\n📝 HEADING FORMAT CHECK:")
        print(f"  Hash count: {hash_count}")
        if hash_count >= 1:
            print(f"  ✅ Properly formatted as markdown heading")
        else:
            print(f"  ❌ Not formatted as markdown heading")

        assert hash_count >= 1, f"Title not properly formatted as heading: {title_line}"

        print(f"\n🎉 TITLE EXTRACTION TEST: ✅ PASSED")
        print("="*60)

    def test_section_structure(self, markdown_content):
        """Test that all major paper sections are extracted"""
        print("\n" + "="*60)
        print("SECTION STRUCTURE TEST - DETAILED ANALYSIS")
        print("="*60)

        sections = extract_sections(markdown_content)
        print(f"🏗️  SECTION DETECTION:")
        print(f"  Found {len(sections)} total sections")

        if not sections:
            print("  ❌ ERROR: No sections found in markdown")
            assert False, "No sections found in markdown"
        else:
            print(f"  ✅ Sections detected")

        # Show all detected sections with levels
        print(f"\n📋 DETECTED SECTIONS:")
        for section in sections:
            level_indicator = "#" * section['level']
            print(f"  {level_indicator} {section['title']} (Level {section['level']})")

        required_sections = [
            "Abstract",
            "Introduction",
            "Neural Galerkin",
            "Randomized Sparse",
            "Experiments",
            "Results",
            "Conclusion",
            "References"
        ]

        print(f"\n🎯 REQUIRED SECTIONS ANALYSIS:")
        missing_sections = []
        found_sections = []
        level_issues = []

        for section in required_sections:
            if section_exists(sections, section):
                found_sections.append(section)
                # Get the actual section info
                actual_section = next(s for s in sections if section.lower() in s['title'].lower())
                level = actual_section['level']

                if proper_heading_level(sections, section):
                    print(f"  ✅ '{section}': Found (Level {level}) - Good hierarchy")
                else:
                    print(f"  ⚠️  '{section}': Found (Level {level}) - Hierarchy issue")
                    level_issues.append(f"{section} (Level {level})")
            else:
                missing_sections.append(section)
                print(f"  ❌ '{section}': Not found")

        print(f"\n📊 SECTION STRUCTURE SUMMARY:")
        print(f"  Required sections: {len(required_sections)}")
        print(f"  Found sections: {len(found_sections)}/{len(required_sections)}")
        print(f"  Missing sections: {missing_sections}")
        print(f"  Heading level issues: {level_issues}")

        # Check for hierarchy problems
        if level_issues:
            print(f"\n⚠️  HEADING HIERARCHY ISSUES DETECTED:")
            for issue in level_issues:
                print(f"    - {issue}")
            print(f"  Expected hierarchy: H1 (title) → H2 (main sections) → H3 (subsections)")

        if not missing_sections:
            print(f"  ✅ All required sections found")
        else:
            print(f"  ❌ Missing {len(missing_sections)} required sections")

        if not level_issues:
            print(f"  ✅ Heading hierarchy is proper")
        else:
            print(f"  ❌ {len(level_issues)} heading level issues found")

        assert not missing_sections, f"Missing required sections: {missing_sections}"

        # Check heading levels are reasonable
        for section in required_sections:
            if section_exists(sections, section):
                assert proper_heading_level(sections, section), f"Section '{section}' has improper heading level"

        print(f"\n🎉 SECTION STRUCTURE TEST: ✅ PASSED")
        print("="*60)

    def test_mathematical_content_preservation(self, markdown_content):
        """Test that mathematical equations and formulas are preserved"""
        print("\n" + "="*60)
        print("MATHEMATICAL CONTENT PRESERVATION TEST - DETAILED ANALYSIS")
        print("="*60)

        # Check for mathematical notation preservation
        math_indicators = [
            "$",  # Inline math
            "$$", # Block math
            "\\begin{equation}",  # LaTeX equations
            "\\mathbf",  # Bold math symbols
            "\\partial",  # Partial derivatives
            "\\theta"  # Neural network parameters
        ]

        print(f"🔢 MATHEMATICAL NOTATION ANALYSIS:")
        print(f"Searching for {len(math_indicators)} types of mathematical notation...")

        found_indicators = []
        for indicator in math_indicators:
            count = markdown_content.count(indicator)
            if count > 0:
                found_indicators.append(indicator)
                print(f"  ✅ '{indicator}': Found {count} occurrence(s)")
            else:
                print(f"  ❌ '{indicator}': Not found")

        math_content_found = len(found_indicators)
        print(f"\n📊 MATHEMATICAL NOTATION SUMMARY: {math_content_found}/{len(math_indicators)} types found")

        if math_content_found >= 3:
            print(f"  ✅ Sufficient mathematical notation preserved (≥3 required)")
        else:
            print(f"  ❌ Insufficient mathematical notation (need ≥3, found {math_content_found})")

        assert math_content_found >= 3, f"Insufficient mathematical notation found: {math_content_found}/6 indicators"

        # Check specific mathematical concepts from the paper
        mathematical_concepts = [
            "Galerkin projection",
            "sparse update",
            "randomization",
            "time evolution",
            "parameter subset",
            "neural network",
            "variational"
        ]

        print(f"\n🧠 MATHEMATICAL CONCEPTS ANALYSIS:")
        print(f"Checking for {len(mathematical_concepts)} key mathematical concepts...")

        missing_concepts = []
        found_concepts = []
        for concept in mathematical_concepts:
            if concept.lower() in markdown_content.lower():
                found_concepts.append(concept)
                # Find and show context
                lines = markdown_content.split('\n')
                context_lines = []
                for line in lines:
                    if concept.lower() in line.lower():
                        context_lines.append(line.strip()[:80] + '...' if len(line.strip()) > 80 else line.strip())
                        break  # Just show first occurrence
                context = context_lines[0] if context_lines else "Context not found"
                print(f"  ✅ '{concept}': Found - {context}")
            else:
                missing_concepts.append(concept)
                print(f"  ❌ '{concept}': Not found")

        print(f"\n📊 MATHEMATICAL CONCEPTS SUMMARY:")
        print(f"  Found: {len(found_concepts)}/{len(mathematical_concepts)} concepts")
        print(f"  Missing: {missing_concepts}")

        concepts_threshold = len(mathematical_concepts) - 2  # Allow up to 2 missing
        if len(found_concepts) >= concepts_threshold:
            print(f"  ✅ Sufficient concepts preserved ({len(found_concepts)}/{len(mathematical_concepts)}, ≥{concepts_threshold} required)")
        else:
            print(f"  ❌ Too many concepts missing ({len(found_concepts)}/{len(mathematical_concepts)}, ≥{concepts_threshold} required)")

        assert len(missing_concepts) <= 2, f"Too many mathematical concepts missing: {missing_concepts}"

        print(f"\n🎉 MATHEMATICAL CONTENT TEST: ✅ PASSED")
        print("="*60)

    def test_algorithm_extraction(self, markdown_content):
        """Test that algorithms and pseudocode are properly formatted"""
        print("\n" + "="*60)
        print("ALGORITHM EXTRACTION TEST - DETAILED ANALYSIS")
        print("="*60)

        # Look for algorithm blocks
        algorithm_content = extract_algorithm_blocks(markdown_content)
        print(f"🔍 ALGORITHM DETECTION:")
        print(f"  Found {len(algorithm_content)} potential algorithm block(s)")

        if len(algorithm_content) == 0:
            print("  ❌ ERROR: No algorithms found in markdown")
            assert False, "No algorithms found in markdown"
        else:
            print(f"  ✅ Algorithm blocks detected")

        # Check algorithm formatting
        properly_formatted = 0
        contains_steps = 0

        print(f"\n📋 ALGORITHM ANALYSIS:")
        for i, algorithm in enumerate(algorithm_content, 1):
            print(f"  Algorithm {i}:")

            # Show preview
            preview = algorithm[:100].replace('\n', ' ').strip()
            print(f"    Preview: '{preview}{'...' if len(algorithm) > 100 else ''}'")

            # Check formatting
            is_formatted = is_properly_formatted_code_block(algorithm)
            has_steps = contains_key_steps(algorithm)

            if is_formatted:
                properly_formatted += 1
                print(f"    ✅ Properly formatted as code block")
            else:
                print(f"    ❌ Not properly formatted as code block")

            if has_steps:
                contains_steps += 1
                # Show detected step indicators
                step_indicators = ['step', 'for', 'while', 'if', 'return', 'compute', 'update',
                                  'algorithm', 'procedure', 'function', 'input:', 'output:']
                found_indicators = [ind for ind in step_indicators if ind in algorithm.lower()]
                print(f"    ✅ Contains procedural steps: {found_indicators[:3]}{'...' if len(found_indicators) > 3 else ''}")
            else:
                print(f"    ❌ No clear procedural steps detected")

        print(f"\n📊 ALGORITHM FORMATTING SUMMARY:")
        print(f"  Total algorithms: {len(algorithm_content)}")
        print(f"  Properly formatted: {properly_formatted}/{len(algorithm_content)}")
        print(f"  Contains steps: {contains_steps}/{len(algorithm_content)}")

        if properly_formatted >= 1:
            print(f"  ✅ Sufficient properly formatted algorithms (≥1 required)")
        else:
            print(f"  ❌ No properly formatted algorithm blocks found")

        if contains_steps >= 1:
            print(f"  ✅ Sufficient algorithms with steps (≥1 required)")
        else:
            print(f"  ❌ No algorithms with procedural steps found")

        assert properly_formatted >= 1, f"No properly formatted algorithm blocks found (found {properly_formatted})"
        assert contains_steps >= 1, f"No algorithms with procedural steps found (found {contains_steps})"

        print(f"\n🎉 ALGORITHM EXTRACTION TEST: ✅ PASSED")
        print("="*60)

    def test_references_and_citations(self, markdown_content):
        """Test that references and citations are preserved"""
        print("\n" + "="*60)
        print("REFERENCES AND CITATIONS TEST - DETAILED ANALYSIS")
        print("="*60)

        # Check References section exists
        has_references = ("references" in markdown_content.lower() or
                         "bibliography" in markdown_content.lower())
        print(f"📚 REFERENCES SECTION DETECTION:")
        if has_references:
            print(f"  ✅ References section found")
        else:
            print(f"  ❌ ERROR: No references section found")
            assert False, "No references section found"

        # Check reference format is preserved
        reference_section = extract_references_section(markdown_content)
        print(f"  Section length: {len(reference_section)} characters")

        if not reference_section.strip():
            print(f"  ❌ ERROR: References section appears to be empty")
            assert False, "References section appears to be empty"
        else:
            print(f"  ✅ References section contains content")

        references = parse_references(reference_section)
        print(f"\n🔍 REFERENCE PARSING:")
        print(f"  Found {len(references)} individual references")

        if len(references) >= 5:
            print(f"  ✅ Sufficient references (≥5 required)")
        else:
            print(f"  ❌ Too few references (need ≥5, found {len(references)})")

        # Show sample references
        print(f"\n📋 SAMPLE REFERENCES:")
        for i, ref in enumerate(references[:3], 1):
            preview = ref.replace('\n', ' ').strip()[:100]
            print(f"  [{i}] {preview}{'...' if len(preview) >= 100 else ''}")

        assert len(references) >= 5, f"Too few references found: {len(references)}, expected at least 5"

        # Check reference formatting
        refs_with_authors = sum(1 for ref in references if has_author_info(ref))
        refs_with_pub_info = sum(1 for ref in references if has_publication_info(ref))

        print(f"\n📊 REFERENCE QUALITY ANALYSIS:")
        print(f"  Total references: {len(references)}")
        print(f"  With author info: {refs_with_authors}/{len(references)} ({refs_with_authors/len(references)*100:.1f}%)")
        print(f"  With publication info: {refs_with_pub_info}/{len(references)} ({refs_with_pub_info/len(references)*100:.1f}%)")

        author_threshold = len(references) * 0.7
        pub_threshold = len(references) * 0.5

        if refs_with_authors >= author_threshold:
            print(f"  ✅ Sufficient author information (≥70% required)")
        else:
            print(f"  ❌ Insufficient author information (need ≥{author_threshold:.1f}, found {refs_with_authors})")

        if refs_with_pub_info >= pub_threshold:
            print(f"  ✅ Sufficient publication information (≥50% required)")
        else:
            print(f"  ❌ Insufficient publication information (need ≥{pub_threshold:.1f}, found {refs_with_pub_info})")

        # Show references with issues
        print(f"\n🔍 REFERENCE DETAILS:")
        for i, ref in enumerate(references, 1):
            has_author = has_author_info(ref)
            has_pub = has_publication_info(ref)
            status = "✅" if (has_author and has_pub) else "⚠️" if (has_author or has_pub) else "❌"
            issues = []
            if not has_author: issues.append("No author")
            if not has_pub: issues.append("No pub info")
            issue_str = f" ({', '.join(issues)})" if issues else ""
            preview = ref.replace('\n', ' ').strip()[:60]
            print(f"  {status} [{i}] {preview}{'...' if len(preview) >= 60 else ''}{issue_str}")

        assert refs_with_authors >= author_threshold, f"Too few references have author info: {refs_with_authors}/{len(references)} (need ≥70%)"
        assert refs_with_pub_info >= pub_threshold, f"Too few references have publication info: {refs_with_pub_info}/{len(references)} (need ≥50%)"

        # Check in-text citations exist
        citation_patterns = [r'\[(\d+)\]', r'\[\d+,\s*\d+\]', r'\[\d+-\d+\]', r'\(\d{4}\)']
        print(f"\n📝 IN-TEXT CITATIONS ANALYSIS:")

        total_citations = 0
        for pattern in citation_patterns:
            count = count_citation_patterns(markdown_content, [pattern])
            total_citations += count
            pattern_desc = {
                r'\[(\d+)\]': 'Single citations [1]',
                r'\[\d+,\s*\d+\]': 'Multiple citations [1,2]',
                r'\[\d+-\d+\]': 'Range citations [1-3]',
                r'\(\d{4}\)': 'Year citations (2020)'
            }
            print(f"  {pattern_desc.get(pattern, pattern)}: {count} found")

        print(f"  Total citations: {total_citations}")

        if total_citations >= 3:
            print(f"  ✅ Sufficient in-text citations (≥3 required)")
        else:
            print(f"  ❌ Too few in-text citations (need ≥3, found {total_citations})")

        assert total_citations >= 3, f"Too few in-text citations found: {total_citations}"

        print(f"\n🎉 REFERENCES AND CITATIONS TEST: ✅ PASSED")
        print("="*60)

    def test_markdown_quality(self, markdown_content):
        """Test overall markdown quality and readability"""
        print("\n" + "="*60)
        print("MARKDOWN QUALITY TEST - DETAILED ANALYSIS")
        print("="*60)

        quality_checks = {
            'proper_heading_hierarchy': check_heading_hierarchy(markdown_content),
            'no_broken_links': verify_no_broken_links(markdown_content),
            'consistent_formatting': check_formatting_consistency(markdown_content),
            'readable_tables': verify_table_formatting(markdown_content),
            'image_references': check_image_references(markdown_content)
        }

        print(f"🔍 MARKDOWN QUALITY CHECKS:")
        passed_checks = []
        failed_checks = []

        for check_name, result in quality_checks.items():
            check_display_name = check_name.replace('_', ' ').title()
            if result:
                passed_checks.append(check_name)
                print(f"  ✅ {check_display_name}: PASSED")
            else:
                failed_checks.append(check_name)
                print(f"  ❌ {check_display_name}: FAILED")

        print(f"\n📊 QUALITY SUMMARY:")
        print(f"  Total checks: {len(quality_checks)}")
        print(f"  Passed: {len(passed_checks)}/{len(quality_checks)} ({len(passed_checks)/len(quality_checks)*100:.1f}%)")
        print(f"  Failed: {len(failed_checks)}/{len(quality_checks)} ({len(failed_checks)/len(quality_checks)*100:.1f}%)")

        if failed_checks:
            print(f"\n⚠️  FAILED CHECKS DETAILS:")
            for check in failed_checks:
                if check == 'proper_heading_hierarchy':
                    print(f"    - Heading Hierarchy: Inconsistent heading levels detected")
                elif check == 'no_broken_links':
                    print(f"    - Broken Links: Links that don't resolve properly found")
                elif check == 'consistent_formatting':
                    print(f"    - Consistent Formatting: Formatting inconsistencies detected")
                elif check == 'readable_tables':
                    print(f"    - Readable Tables: No properly formatted markdown tables found")
                elif check == 'image_references':
                    print(f"    - Image References: Issues with image formatting or alt text")

        if not failed_checks:
            print(f"  ✅ All quality checks passed")
        else:
            print(f"  ❌ {len(failed_checks)} quality issues need attention")

        assert not failed_checks, f"Quality checks failed: {failed_checks}"

        print(f"\n🎉 MARKDOWN QUALITY TEST: ✅ PASSED")
        print("="*60)

    def test_completeness_metrics(self, markdown_content):
        """Test that conversion captured reasonable percentage of content"""
        print("\n" + "="*60)
        print("COMPLETENESS METRICS TEST - DETAILED ANALYSIS")
        print("="*60)

        # Rough metrics for completeness
        word_count = count_words(markdown_content)
        print(f"📊 DOCUMENT SIZE ANALYSIS:")
        print(f"  Word count: {word_count:,} words")

        # Check word count ranges
        if 2000 <= word_count <= 20000:
            print(f"  ✅ Word count within expected range (2,000-20,000)")
        elif word_count < 2000:
            print(f"  ❌ Document too short (minimum 2,000 words)")
        else:
            print(f"  ❌ Document suspiciously long (maximum 20,000 words)")

        assert word_count >= 2000, f"Paper seems too short: {word_count} words (expected 2000+)"
        assert word_count <= 20000, f"Paper seems suspiciously long: {word_count} words (expected <20000)"

        # Check content distribution
        content_metrics = analyze_content_distribution(markdown_content)
        print(f"\n📈 CONTENT DISTRIBUTION ANALYSIS:")
        print(f"  Text percentage: {content_metrics['text_percentage']:.1f}%")
        print(f"  Mathematical content: {content_metrics['math_percentage']:.1f}%")

        # Text content validation
        text_threshold = 60
        if content_metrics['text_percentage'] >= text_threshold:
            print(f"  ✅ Sufficient text content (≥{text_threshold}% required)")
        else:
            print(f"  ❌ Insufficient text content (need ≥{text_threshold}%, found {content_metrics['text_percentage']:.1f}%)")

        # Mathematical content validation
        math_threshold = 2
        if content_metrics['math_percentage'] >= math_threshold:
            print(f"  ✅ Sufficient mathematical content (≥{math_threshold}% required)")
        else:
            print(f"  ❌ Insufficient mathematical content (need ≥{math_threshold}%, found {content_metrics['math_percentage']:.1f}%)")

        # Show additional metrics if available
        if 'section_count' in content_metrics:
            print(f"  Section count: {content_metrics['section_count']}")
        if 'average_section_length' in content_metrics:
            print(f"  Average section length: {content_metrics['average_section_length']:.0f} words")

        print(f"\n📋 COMPLETENESS SUMMARY:")
        if (word_count >= 2000 and content_metrics['text_percentage'] >= 60 and content_metrics['math_percentage'] >= 2):
            print(f"  ✅ Document appears complete and well-structured")
        else:
            issues = []
            if word_count < 2000: issues.append("too short")
            if content_metrics['text_percentage'] < 60: issues.append("insufficient text")
            if content_metrics['math_percentage'] < 2: issues.append("insufficient math")
            print(f"  ❌ Completeness issues: {', '.join(issues)}")

        assert content_metrics['text_percentage'] >= 60, f"Too little text content: {content_metrics['text_percentage']:.1f}%"
        assert content_metrics['math_percentage'] >= 2, f"Too little mathematical content: {content_metrics['math_percentage']:.1f}%"

        print(f"\n🎉 COMPLETENESS METRICS TEST: ✅ PASSED")
        print("="*60)

    def test_technical_term_preservation(self, markdown_content):
        """Test that technical terms are preserved correctly"""
        print("\n" + "="*60)
        print("TECHNICAL TERM PRESERVATION TEST - DETAILED ANALYSIS")
        print("="*60)

        critical_terms = [
            "Neural Galerkin schemes",
            "randomized sparse subsets",
            "evolution equations",
            "sequential-in-time training",
            "parameter updates",
            "time-dependent PDEs",
            "variational principle",
            "overfitting"
        ]

        print(f"🔬 TECHNICAL TERMINOLOGY ANALYSIS:")
        print(f"Checking {len(critical_terms)} critical technical terms...")

        missing_terms = []
        improperly_contextualized = []
        found_terms = []

        for term in critical_terms:
            if term not in markdown_content:
                missing_terms.append(term)
                print(f"  ❌ '{term}': Not found")
            elif not appears_in_proper_context(markdown_content, term):
                improperly_contextualized.append(term)
                print(f"  ⚠️  '{term}': Found but context unclear")
            else:
                found_terms.append(term)
                # Find and show context
                lines = markdown_content.split('\n')
                for line in lines:
                    if term in line:
                        context = line.strip()[:100]
                        print(f"  ✅ '{term}': Found - {context}{'...' if len(context) >= 100 else ''}")
                        break

        print(f"\n📊 TECHNICAL TERMS SUMMARY:")
        print(f"  Total terms checked: {len(critical_terms)}")
        print(f"  Found and properly contextualized: {len(found_terms)}")
        print(f"  Found but context unclear: {len(improperly_contextualized)}")
        print(f"  Missing: {len(missing_terms)}")

        if missing_terms:
            print(f"\n❌ MISSING TERMS:")
            for term in missing_terms:
                print(f"    - {term}")

        if improperly_contextualized:
            print(f"\n⚠️  CONTEXT ISSUES:")
            for term in improperly_contextualized:
                print(f"    - {term}")

        # Thresholds
        max_missing = 2
        max_context_issues = 1

        print(f"\n🎯 VALIDATION THRESHOLDS:")
        print(f"  Maximum missing terms allowed: {max_missing}")
        print(f"  Maximum context issues allowed: {max_context_issues}")

        if len(missing_terms) <= max_missing:
            print(f"  ✅ Missing terms within acceptable range ({len(missing_terms)}/{max_missing})")
        else:
            print(f"  ❌ Too many missing terms ({len(missing_terms)}/{max_missing})")

        if len(improperly_contextualized) <= max_context_issues:
            print(f"  ✅ Context issues within acceptable range ({len(improperly_contextualized)}/{max_context_issues})")
        else:
            print(f"  ❌ Too many context issues ({len(improperly_contextualized)}/{max_context_issues})")

        assert len(missing_terms) <= max_missing, f"Critical terms missing: {missing_terms}"
        assert len(improperly_contextualized) <= max_context_issues, f"Terms not in proper context: {improperly_contextualized}"

        print(f"\n🎉 TECHNICAL TERM PRESERVATION TEST: ✅ PASSED")
        print("="*60)


# Test execution function for reporting
def run_extraction_tests():
    """Execute all conversion tests and report results"""
    import subprocess
    import sys

    print("=== PDF to Markdown Extraction Test Results ===")
    print("RED PHASE: All tests should FAIL initially")
    print()

    try:
        # Run pytest with verbose output
        result = subprocess.run([
            sys.executable, '-m', 'pytest',
            '/Users/jacobmccarran_ax/Downloads/nueral-galerkin-casestudy/test_pdf_extraction.py',
            '-v', '--tb=short'
        ], capture_output=True, text=True, cwd='/Users/jacobmccarran_ax/Downloads/nueral-galerkin-casestudy')

        print("STDOUT:")
        print(result.stdout)
        print("\nSTDERR:")
        print(result.stderr)
        print(f"\nReturn code: {result.returncode}")

        return result.returncode == 0

    except Exception as e:
        print(f"Error running tests: {e}")
        return False


if __name__ == "__main__":
    # Run tests when script is executed directly
    success = run_extraction_tests()
    if not success:
        print("\n✅ RED PHASE COMPLETE: All tests failed as expected!")
        print("Next step: Execute PDF to Markdown conversion (GREEN phase)")
    else:
        print("\n❌ Unexpected: Some tests passed - markdown file may already exist")