"""
Web-based Epitope Validation with Mandatory Citations

This module validates predicted epitopes by:
1. Searching scientific literature (PubMed, Google Scholar)
2. Checking IEDB database for experimentally validated epitopes
3. Finding structural evidence (PDB database)
4. **MANDATORY**: Citing all sources with DOI/PMID

Every validation MUST include:
- Primary references (papers, databases)
- DOI or PMID identifiers
- Date accessed
- Confidence level based on evidence quality

Usage:
    validator = WebEpitopeValidator()
    result = validator.validate_with_citations(
        epitope_sequence="YQAGSTPCNGVEG",
        antigen_name="spike protein",
        organism="SARS-CoV-2"
    )
"""

import json
import time
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path


@dataclass
class Citation:
    """
    Mandatory citation for every piece of evidence

    All fields are REQUIRED - no validation without proper citation
    """
    source_type: str  # 'pubmed', 'iedb', 'pdb', 'web', 'database'
    title: str
    authors: str
    year: int
    identifier: str  # DOI, PMID, PDB ID, or URL
    url: str
    accessed_date: str  # ISO format
    relevant_text: str = ""  # Quote from source

    def format_citation(self) -> str:
        """Format as publication-ready citation"""
        if self.source_type == 'pubmed':
            return f"{self.authors} ({self.year}). {self.title}. PMID: {self.identifier}. {self.url}"
        elif self.source_type == 'pdb':
            return f"Protein Data Bank entry {self.identifier}: {self.title} ({self.year}). {self.url}"
        elif self.source_type == 'iedb':
            return f"IEDB Epitope ID {self.identifier}: {self.title}. {self.url} (Accessed: {self.accessed_date})"
        else:
            return f"{self.authors} ({self.year}). {self.title}. {self.url} (Accessed: {self.accessed_date})"


@dataclass
class ValidationEvidence:
    """
    Evidence for epitope validation

    Every piece of evidence MUST have citation(s)
    """
    evidence_type: str  # 'experimental', 'structural', 'computational', 'database'
    description: str
    confidence_level: str  # 'high', 'medium', 'low'
    citations: List[Citation]  # MANDATORY - must have at least one
    validation_method: str = ""  # e.g., "ELISA", "X-ray crystallography"

    def __post_init__(self):
        if not self.citations:
            raise ValueError("❌ VALIDATION ERROR: Evidence must have at least one citation!")


@dataclass
class EpitopeValidationResult:
    """
    Complete validation result with mandatory citations
    """
    epitope_sequence: str
    start_position: int
    end_position: int
    antigen_name: str
    organism: str

    # Validation status
    is_validated: bool
    validation_confidence: str  # 'high', 'medium', 'low', 'none'
    validation_summary: str

    # Evidence with citations (MANDATORY if validated)
    evidence: List[ValidationEvidence]

    # All citations collected
    all_citations: List[Citation]

    # Metadata
    validation_date: str
    validation_method: str

    def get_citation_count(self) -> int:
        """Count total citations"""
        return len(self.all_citations)

    def has_experimental_evidence(self) -> bool:
        """Check if has experimental validation"""
        return any(e.evidence_type == 'experimental' for e in self.evidence)

    def has_structural_evidence(self) -> bool:
        """Check if has structural evidence (PDB)"""
        return any(e.evidence_type == 'structural' for e in self.evidence)

    def generate_citation_report(self) -> str:
        """Generate formatted citation report"""
        report = f"## Citations for {self.epitope_sequence}\n\n"
        report += f"**Total Citations**: {self.get_citation_count()}\n\n"

        for i, citation in enumerate(self.all_citations, 1):
            report += f"{i}. {citation.format_citation()}\n\n"

        return report


class WebEpitopeValidator:
    """
    Validates epitopes using web search with MANDATORY citations

    This validator enforces strict citation requirements:
    - Every claim must have a source
    - Every source must have DOI/PMID/URL
    - Confidence based on citation quality

    Now uses REAL API integrations!
    """

    def __init__(self,
                 require_citations: bool = True,
                 min_citations: int = 1,
                 email: Optional[str] = None,
                 ncbi_api_key: Optional[str] = None):
        """
        Args:
            require_citations: If True, reject validation without citations (default: True)
            min_citations: Minimum citations required for validation (default: 1)
            email: Email for NCBI E-utilities (required for PubMed)
            ncbi_api_key: Optional NCBI API key for higher rate limits
        """
        self.require_citations = require_citations
        self.min_citations = min_citations
        self.validation_date = datetime.now().isoformat()

        # Initialize integrated validator if email provided
        self.integrated_validator = None
        if email:
            try:
                from api_integrations import IntegratedValidator
                self.integrated_validator = IntegratedValidator(email, ncbi_api_key)
                print("✅ Real API integrations enabled (PubMed, IEDB, PDB)")
            except ImportError as e:
                print(f"⚠️  Could not load API integrations: {e}")
                print("   Falling back to placeholder mode")
        else:
            print("⚠️  No email provided - API integrations disabled")
            print("   Provide email to enable PubMed, IEDB, and PDB searches")

    def validate_with_citations(self,
                                epitope_sequence: str,
                                antigen_name: str,
                                organism: str,
                                start_position: int = 0,
                                end_position: int = 0) -> EpitopeValidationResult:
        """
        Validate epitope with mandatory citations

        This method:
        1. Searches PubMed for papers mentioning the epitope
        2. Checks IEDB for experimental validation
        3. Searches PDB for structural evidence
        4. Compiles all citations

        Args:
            epitope_sequence: Epitope amino acid sequence
            antigen_name: Name of antigen
            organism: Organism name
            start_position: Start position in antigen
            end_position: End position in antigen

        Returns:
            EpitopeValidationResult with citations
        """
        print(f"\n{'='*80}")
        print(f"🔍 VALIDATING EPITOPE WITH MANDATORY CITATIONS")
        print(f"{'='*80}")
        print(f"Epitope: {epitope_sequence}")
        print(f"Antigen: {antigen_name} ({organism})")
        print(f"Position: {start_position}-{end_position}")
        print(f"{'='*80}\n")

        # Collect all evidence
        all_evidence = []
        all_citations = []

        # Use real API integrations if available
        if self.integrated_validator:
            print("🔗 Using real API integrations (PubMed, IEDB, PDB)")
            evidence_list = self.integrated_validator.validate_epitope(
                epitope_sequence, antigen_name, organism
            )
            all_evidence.extend(evidence_list)
            for ev in evidence_list:
                all_citations.extend(ev.citations)
        else:
            # Fallback to placeholder methods
            print("⚠️  Using placeholder methods (no email provided)")

            # 1. Search PubMed
            print("📚 Step 1: Searching PubMed...")
            pubmed_evidence = self._search_pubmed(epitope_sequence, antigen_name, organism)
            if pubmed_evidence:
                all_evidence.extend(pubmed_evidence)
                for ev in pubmed_evidence:
                    all_citations.extend(ev.citations)

            # 2. Search IEDB
            print("\n🔬 Step 2: Searching IEDB database...")
            iedb_evidence = self._search_iedb(epitope_sequence, antigen_name, organism)
            if iedb_evidence:
                all_evidence.extend(iedb_evidence)
                for ev in iedb_evidence:
                    all_citations.extend(ev.citations)

            # 3. Search PDB
            print("\n🧬 Step 3: Searching PDB for structures...")
            pdb_evidence = self._search_pdb(epitope_sequence, antigen_name, organism)
            if pdb_evidence:
                all_evidence.extend(pdb_evidence)
                for ev in pdb_evidence:
                    all_citations.extend(ev.citations)

            # 4. Web search for additional evidence
            print("\n🌐 Step 4: General web search...")
            web_evidence = self._web_search(epitope_sequence, antigen_name, organism)
            if web_evidence:
                all_evidence.extend(web_evidence)
                for ev in web_evidence:
                    all_citations.extend(ev.citations)

        # Determine validation status
        is_validated = len(all_citations) >= self.min_citations
        confidence = self._calculate_confidence(all_evidence)

        # Generate summary
        summary = self._generate_summary(all_evidence, all_citations, is_validated, confidence)

        # Create result
        result = EpitopeValidationResult(
            epitope_sequence=epitope_sequence,
            start_position=start_position,
            end_position=end_position,
            antigen_name=antigen_name,
            organism=organism,
            is_validated=is_validated,
            validation_confidence=confidence,
            validation_summary=summary,
            evidence=all_evidence,
            all_citations=all_citations,
            validation_date=self.validation_date,
            validation_method="Web-based with mandatory citations"
        )

        # Print summary
        print(f"\n{'='*80}")
        print("📊 VALIDATION SUMMARY")
        print(f"{'='*80}")
        print(f"Status: {'✅ VALIDATED' if is_validated else '⚠️  NOT VALIDATED'}")
        print(f"Confidence: {confidence.upper()}")
        print(f"Evidence pieces: {len(all_evidence)}")
        print(f"Total citations: {len(all_citations)}")
        print(f"Experimental evidence: {'Yes' if result.has_experimental_evidence() else 'No'}")
        print(f"Structural evidence: {'Yes' if result.has_structural_evidence() else 'No'}")
        print(f"{'='*80}\n")

        if not is_validated and self.require_citations:
            print("⚠️  WARNING: Epitope not validated - insufficient citations")
            print(f"   Required: {self.min_citations}, Found: {len(all_citations)}")

        return result

    def _search_pubmed(self, epitope: str, antigen: str, organism: str) -> List[ValidationEvidence]:
        """
        Search PubMed for papers about the epitope

        NOTE: This is a PLACEHOLDER implementation
        In production, integrate with:
        - NCBI E-utilities API: https://www.ncbi.nlm.nih.gov/books/NBK25501/
        - Bio.Entrez (Biopython)
        """
        print(f"   Query: '{organism} {antigen} epitope {epitope[:10]}'")
        print(f"   ⚠️  PubMed search not yet implemented - using placeholder")

        # PLACEHOLDER: In production, make actual API calls
        # Example query: https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term=...

        evidence = []

        # Placeholder citation
        # In production: Parse actual PubMed results
        placeholder_citation = Citation(
            source_type='pubmed',
            title=f"Identification and characterization of {antigen} epitopes in {organism}",
            authors="Smith J, et al.",
            year=2024,
            identifier="PLACEHOLDER_PMID",
            url="https://pubmed.ncbi.nlm.nih.gov/",
            accessed_date=datetime.now().isoformat(),
            relevant_text="This is a placeholder. Implement actual PubMed search."
        )

        # Don't add placeholder evidence - return empty for now
        # evidence.append(ValidationEvidence(
        #     evidence_type='computational',
        #     description="Placeholder - PubMed search not implemented",
        #     confidence_level='low',
        #     citations=[placeholder_citation],
        #     validation_method="Literature search (placeholder)"
        # ))

        return evidence

    def _search_iedb(self, epitope: str, antigen: str, organism: str) -> List[ValidationEvidence]:
        """
        Search IEDB for experimentally validated epitopes

        NOTE: This is a PLACEHOLDER
        In production, use IEDB API: http://tools.iedb.org/main/tools-api/
        """
        print(f"   Searching IEDB for exact match...")
        print(f"   ⚠️  IEDB search not yet implemented - using placeholder")

        evidence = []

        # PLACEHOLDER: Implement actual IEDB API query
        # Example: http://tools.iedb.org/bcell/result.php?epitope_sequence=...

        # For now, return empty - don't use fake citations
        return evidence

    def _search_pdb(self, epitope: str, antigen: str, organism: str) -> List[ValidationEvidence]:
        """
        Search Protein Data Bank for structures

        NOTE: This is a PLACEHOLDER
        In production, use PDB API: https://www.rcsb.org/docs/programmatic-access
        """
        print(f"   Searching PDB for antibody-antigen complexes...")
        print(f"   ⚠️  PDB search not yet implemented - using placeholder")

        evidence = []

        # PLACEHOLDER: Implement actual PDB search
        # Example: https://search.rcsb.org/rcsbsearch/v2/query?json=...

        return evidence

    def _web_search(self, epitope: str, antigen: str, organism: str) -> List[ValidationEvidence]:
        """
        General web search for epitope mentions

        NOTE: This is a PLACEHOLDER
        In production, use web search API or WebSearch tool
        """
        print(f"   Searching web for general mentions...")
        print(f"   ⚠️  Web search not yet implemented - using placeholder")

        evidence = []

        # PLACEHOLDER: Would use WebSearch tool here
        # For now, return empty

        return evidence

    def _calculate_confidence(self, evidence: List[ValidationEvidence]) -> str:
        """
        Calculate confidence level based on evidence quality

        High: Experimental validation + structural evidence
        Medium: Computational prediction + database evidence
        Low: Only web mentions
        None: No evidence
        """
        if not evidence:
            return 'none'

        has_experimental = any(e.evidence_type == 'experimental' for e in evidence)
        has_structural = any(e.evidence_type == 'structural' for e in evidence)
        has_database = any(e.evidence_type == 'database' for e in evidence)

        if has_experimental and has_structural:
            return 'high'
        elif has_experimental or has_structural or has_database:
            return 'medium'
        elif evidence:
            return 'low'
        else:
            return 'none'

    def _generate_summary(self,
                         evidence: List[ValidationEvidence],
                         citations: List[Citation],
                         is_validated: bool,
                         confidence: str) -> str:
        """Generate human-readable summary"""

        if not is_validated:
            return (
                f"⚠️  This epitope could not be validated due to insufficient published evidence. "
                f"Found {len(citations)} citations. Minimum required: {self.min_citations}. "
                f"This does not mean the epitope is incorrect, only that it lacks experimental validation in literature. "
                f"Consider this a NOVEL predicted epitope requiring experimental validation."
            )

        summary_parts = [
            f"✅ This epitope has been validated with {len(citations)} citation(s) from scientific literature."
        ]

        # Add evidence types
        evidence_types = list(set(e.evidence_type for e in evidence))
        summary_parts.append(f"Evidence types: {', '.join(evidence_types)}.")

        # Add confidence explanation
        if confidence == 'high':
            summary_parts.append(
                "HIGH confidence due to experimental validation and/or structural evidence."
            )
        elif confidence == 'medium':
            summary_parts.append(
                "MEDIUM confidence based on computational predictions or database entries."
            )
        else:
            summary_parts.append(
                "LOW confidence - limited evidence available. Further validation recommended."
            )

        return " ".join(summary_parts)

    def batch_validate(self,
                      epitopes: List[Dict[str, Any]],
                      output_file: Optional[Path] = None) -> List[EpitopeValidationResult]:
        """
        Validate multiple epitopes

        Args:
            epitopes: List of epitope dicts with 'sequence', 'antigen_name', 'organism', etc.
            output_file: Optional JSON file to save results

        Returns:
            List of validation results
        """
        results = []

        for i, epitope in enumerate(epitopes):
            print(f"\n{'#'*80}")
            print(f"VALIDATING EPITOPE {i+1}/{len(epitopes)}")
            print(f"{'#'*80}")

            result = self.validate_with_citations(
                epitope_sequence=epitope['sequence'],
                antigen_name=epitope.get('antigen_name', 'Unknown'),
                organism=epitope.get('organism', 'Unknown'),
                start_position=epitope.get('start_position', 0),
                end_position=epitope.get('end_position', 0)
            )

            results.append(result)

            # Rate limiting
            time.sleep(2)

        # Save if requested
        if output_file:
            self._save_validation_results(results, output_file)

        return results

    def _save_validation_results(self, results: List[EpitopeValidationResult], output_file: Path):
        """Save validation results to JSON"""
        output_data = {
            'validation_date': self.validation_date,
            'total_epitopes': len(results),
            'validated_count': sum(1 for r in results if r.is_validated),
            'results': [self._result_to_dict(r) for r in results]
        }

        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"\n💾 Saved validation results to {output_file}")

    def _result_to_dict(self, result: EpitopeValidationResult) -> Dict[str, Any]:
        """Convert result to dictionary"""
        return {
            'epitope_sequence': result.epitope_sequence,
            'position': f"{result.start_position}-{result.end_position}",
            'antigen': result.antigen_name,
            'organism': result.organism,
            'is_validated': result.is_validated,
            'confidence': result.validation_confidence,
            'summary': result.validation_summary,
            'evidence_count': len(result.evidence),
            'citation_count': len(result.all_citations),
            'has_experimental': result.has_experimental_evidence(),
            'has_structural': result.has_structural_evidence(),
            'citations': [asdict(c) for c in result.all_citations]
        }


# Example usage
if __name__ == '__main__':
    # Example: Validate a known SARS-CoV-2 epitope
    validator = WebEpitopeValidator(require_citations=True, min_citations=1)

    result = validator.validate_with_citations(
        epitope_sequence="YQAGSTPCNGVEG",
        antigen_name="spike protein",
        organism="SARS-CoV-2",
        start_position=505,
        end_position=518
    )

    print("\n" + "="*80)
    print("VALIDATION RESULT")
    print("="*80)
    print(f"Validated: {result.is_validated}")
    print(f"Confidence: {result.validation_confidence}")
    print(f"Citations: {result.get_citation_count()}")
    print(f"\nSummary:\n{result.validation_summary}")

    if result.all_citations:
        print(f"\n{result.generate_citation_report()}")
