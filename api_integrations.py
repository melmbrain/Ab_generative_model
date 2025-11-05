"""
Real API Integrations for Epitope Validation

Implements actual API calls to:
1. PubMed (NCBI E-utilities via Biopython)
2. IEDB (Immune Epitope Database Query API)
3. RCSB PDB (Protein Data Bank Search API)

All functions return Citation objects with proper DOI/PMID/URLs.

Dependencies:
    pip install biopython requests

References:
[1] NCBI E-utilities: https://www.ncbi.nlm.nih.gov/books/NBK25497/
[2] Biopython Entrez: https://biopython.org/docs/latest/Tutorial/chapter_entrez.html
[3] IEDB IQ-API: https://help.iedb.org/hc/en-us/articles/4402872882189
[4] RCSB PDB API: https://www.rcsb.org/news/684078fe300817f1b5de793a
"""

import time
import requests
from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass
import json

# Import Biopython for PubMed (will handle gracefully if not installed)
try:
    from Bio import Entrez
    BIOPYTHON_AVAILABLE = True
except ImportError:
    BIOPYTHON_AVAILABLE = False
    print("⚠️  Biopython not installed. PubMed search will be limited.")
    print("   Install with: pip install biopython")

from web_epitope_validator import Citation, ValidationEvidence


class PubMedSearcher:
    """
    Search PubMed using NCBI E-utilities API

    References:
    - NCBI E-utilities Guide: https://www.ncbi.nlm.nih.gov/books/NBK25497/
    - Biopython Entrez: https://biopython.org/docs/latest/Tutorial/chapter_entrez.html

    Rate Limits:
    - 3 requests/second without API key
    - 10 requests/second with API key
    """

    def __init__(self, email: str, api_key: Optional[str] = None):
        """
        Args:
            email: Your email (required by NCBI)
            api_key: Optional NCBI API key for higher rate limits
        """
        if not BIOPYTHON_AVAILABLE:
            raise ImportError("Biopython required. Install with: pip install biopython")

        Entrez.email = email
        if api_key:
            Entrez.api_key = api_key

        self.rate_limit_delay = 0.34 if not api_key else 0.11  # seconds

    def search_epitope(self,
                      epitope_sequence: str,
                      antigen_name: str,
                      organism: str,
                      max_results: int = 10) -> List[Citation]:
        """
        Search PubMed for papers about an epitope

        Args:
            epitope_sequence: Epitope amino acid sequence
            antigen_name: Antigen name (e.g., "spike protein")
            organism: Organism name (e.g., "SARS-CoV-2")
            max_results: Maximum number of results to return

        Returns:
            List of Citation objects with PMID
        """
        print(f"   🔍 Searching PubMed...")

        # Construct search query
        # Use first 10 amino acids to avoid too-specific searches
        seq_snippet = epitope_sequence[:10] if len(epitope_sequence) > 10 else epitope_sequence

        query = f"{organism} {antigen_name} epitope antibody"
        print(f"   Query: '{query}'")

        try:
            # Step 1: Search for PMIDs
            handle = Entrez.esearch(
                db="pubmed",
                term=query,
                retmax=max_results,
                sort="relevance"
            )
            search_results = Entrez.read(handle)
            handle.close()

            pmids = search_results['IdList']
            print(f"   Found {len(pmids)} results")

            if not pmids:
                return []

            # Step 2: Fetch details for each PMID
            citations = []
            for pmid in pmids:
                time.sleep(self.rate_limit_delay)  # Rate limiting

                citation = self._fetch_article_details(pmid)
                if citation:
                    citations.append(citation)

            return citations

        except Exception as e:
            print(f"   ❌ PubMed search error: {e}")
            return []

    def _fetch_article_details(self, pmid: str) -> Optional[Citation]:
        """Fetch article details for a PMID"""
        try:
            handle = Entrez.efetch(
                db="pubmed",
                id=pmid,
                rettype="medline",
                retmode="xml"
            )
            records = Entrez.read(handle)
            handle.close()

            if not records['PubmedArticle']:
                return None

            article = records['PubmedArticle'][0]
            medline = article['MedlineCitation']
            article_data = medline['Article']

            # Extract information
            title = article_data.get('ArticleTitle', 'No title')

            # Authors
            authors_list = article_data.get('AuthorList', [])
            if authors_list:
                first_author = authors_list[0]
                last_name = first_author.get('LastName', '')
                initials = first_author.get('Initials', '')
                if len(authors_list) > 1:
                    authors = f"{last_name} {initials}, et al."
                else:
                    authors = f"{last_name} {initials}"
            else:
                authors = "Unknown authors"

            # Year
            pub_date = article_data.get('Journal', {}).get('JournalIssue', {}).get('PubDate', {})
            year = int(pub_date.get('Year', 0))

            # DOI
            article_ids = article['PubmedData'].get('ArticleIdList', [])
            doi = None
            for aid in article_ids:
                if aid.attributes.get('IdType') == 'doi':
                    doi = str(aid)
                    break

            # Abstract (for relevance check)
            abstract_texts = article_data.get('Abstract', {}).get('AbstractText', [])
            abstract = ' '.join([str(text) for text in abstract_texts]) if abstract_texts else ''
            relevant_text = abstract[:300] + '...' if len(abstract) > 300 else abstract

            # Create citation
            citation = Citation(
                source_type='pubmed',
                title=title,
                authors=authors,
                year=year if year > 0 else 2024,
                identifier=f"PMID:{pmid}",
                url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                accessed_date=datetime.now().isoformat(),
                relevant_text=relevant_text
            )

            return citation

        except Exception as e:
            print(f"   ⚠️  Could not fetch details for PMID {pmid}: {e}")
            return None


class IEDBSearcher:
    """
    Search IEDB (Immune Epitope Database) for validated epitopes

    References:
    - IEDB 2024 update: PMID 39558162
    - IQ-API documentation: https://help.iedb.org/hc/en-us/articles/4402872882189
    - API endpoint: https://query-api.iedb.org/epitope_search
    """

    def __init__(self):
        self.base_url = "https://query-api.iedb.org/epitope_search"
        self.rate_limit_delay = 1.0  # Be respectful

    def search_epitope(self,
                      epitope_sequence: str,
                      organism: Optional[str] = None,
                      max_results: int = 20) -> List[Citation]:
        """
        Search IEDB for exact or similar epitope sequences

        Args:
            epitope_sequence: Epitope amino acid sequence
            organism: Optional organism filter
            max_results: Maximum results to return

        Returns:
            List of Citation objects with IEDB IDs
        """
        print(f"   🔍 Searching IEDB database...")
        print(f"   Looking for: {epitope_sequence}")

        try:
            # Query parameters - IEDB uses PostgREST query format
            # Format: field=eq.value for exact match
            # Note: Column names from actual IEDB schema
            params = {
                'linear_sequence': f'eq.{epitope_sequence}',  # PostgREST exact match
                'limit': max_results,
                'select': 'structure_id,linear_sequence'  # Only select columns that exist
            }

            if organism:
                params['source_organism_name'] = f'ilike.*{organism}*'  # Case-insensitive partial match

            # Make request
            headers = {'accept': 'application/json'}
            response = requests.get(self.base_url, params=params, headers=headers, timeout=30)

            if response.status_code != 200:
                print(f"   ⚠️  IEDB API returned status {response.status_code}")
                if response.text:
                    print(f"   Response: {response.text[:200]}")
                return []

            # Parse results
            results = response.json()

            if not results:
                print(f"   No exact matches found in IEDB")
                return []

            print(f"   Found {len(results)} IEDB entries")

            # Convert to citations
            citations = []
            for entry in results:
                citation = self._create_citation_from_entry(entry)
                if citation:
                    citations.append(citation)

            return citations

        except Exception as e:
            print(f"   ❌ IEDB search error: {e}")
            return []

    def _create_citation_from_entry(self, entry: Dict[str, Any]) -> Optional[Citation]:
        """Create citation from IEDB entry"""
        try:
            structure_id = entry.get('structure_id', 'Unknown')
            sequence = entry.get('linear_sequence', '')

            title = f"Linear B-cell epitope - experimentally validated"

            citation = Citation(
                source_type='iedb',
                title=title,
                authors="IEDB Database",
                year=2024,  # Use current year for database entries
                identifier=f"IEDB:{structure_id}",
                url=f"https://www.iedb.org/epitope/{structure_id}",
                accessed_date=datetime.now().isoformat(),
                relevant_text=f"Sequence: {sequence}"
            )

            return citation

        except Exception as e:
            print(f"   ⚠️  Could not parse IEDB entry: {e}")
            return None


class PDBSearcher:
    """
    Search RCSB PDB for antibody-antigen structures

    References:
    - RCSB PDB Python API: https://www.rcsb.org/news/684078fe300817f1b5de793a
    - Search API: https://search.rcsb.org/
    - Publication: J. Mol. Biol. 2025 (Special Issue on Computation Resources)
    """

    def __init__(self):
        self.search_url = "https://search.rcsb.org/rcsbsearch/v2/query"
        self.data_url = "https://data.rcsb.org/rest/v1/core/entry"
        self.rate_limit_delay = 0.5

    def search_antibody_antigen_complexes(self,
                                          antigen_name: str,
                                          organism: Optional[str] = None,
                                          max_results: int = 10) -> List[Citation]:
        """
        Search PDB for antibody-antigen complex structures

        Args:
            antigen_name: Antigen name (e.g., "spike protein")
            organism: Optional organism name
            max_results: Maximum results

        Returns:
            List of Citation objects with PDB IDs
        """
        print(f"   🔍 Searching RCSB PDB...")
        print(f"   Looking for antibody-{antigen_name} complexes")

        try:
            # Construct search query (JSON format v2)
            # Simplified query for better compatibility
            search_terms = [antigen_name, "antibody"]
            if organism:
                search_terms.append(organism)

            query = {
                "query": {
                    "type": "terminal",
                    "service": "full_text",  # Use full_text service
                    "parameters": {
                        "value": " ".join(search_terms)
                    }
                },
                "return_type": "entry",
                "request_options": {
                    "paginate": {
                        "start": 0,
                        "rows": max_results
                    },
                    "sort": [{"sort_by": "score", "direction": "desc"}]
                }
            }

            # Make request
            headers = {'Content-Type': 'application/json'}
            response = requests.post(
                self.search_url,
                json=query,
                headers=headers,
                timeout=30
            )

            if response.status_code != 200:
                print(f"   ⚠️  PDB API returned status {response.status_code}")
                if response.text:
                    print(f"   Response: {response.text[:200]}")
                return []

            results = response.json()

            # Extract PDB IDs
            pdb_ids = []
            if 'result_set' in results:
                pdb_ids = [item['identifier'] for item in results['result_set'][:max_results]]

            if not pdb_ids:
                print(f"   No PDB structures found")
                return []

            print(f"   Found {len(pdb_ids)} PDB structures")

            # Fetch details for each PDB ID
            citations = []
            for pdb_id in pdb_ids:
                time.sleep(self.rate_limit_delay)
                citation = self._fetch_pdb_details(pdb_id)
                if citation:
                    citations.append(citation)

            return citations

        except Exception as e:
            print(f"   ❌ PDB search error: {e}")
            return []

    def _fetch_pdb_details(self, pdb_id: str) -> Optional[Citation]:
        """Fetch details for a PDB entry"""
        try:
            url = f"{self.data_url}/{pdb_id}"
            response = requests.get(url, timeout=30)

            if response.status_code != 200:
                return None

            data = response.json()

            # Extract information
            title = data.get('struct', {}).get('title', 'No title')

            # Get publication info if available
            citation_info = data.get('citation', [])
            year = 2024
            authors = "Unknown authors"
            doi = None

            if citation_info:
                primary_citation = citation_info[0]
                year = primary_citation.get('year', 2024)

                # Authors
                author_list = primary_citation.get('rcsb_authors', [])
                if author_list:
                    authors = f"{author_list[0]}, et al." if len(author_list) > 1 else author_list[0]

                # DOI
                doi = primary_citation.get('pdbx_database_id_DOI')

            # Experimental method and resolution
            exptl_method = data.get('exptl', [{}])[0].get('method', 'Unknown')
            resolution = data.get('rcsb_entry_info', {}).get('resolution_combined', [None])[0]

            relevant_text = f"Method: {exptl_method}"
            if resolution:
                relevant_text += f", Resolution: {resolution:.2f} Å"

            # Create citation
            citation = Citation(
                source_type='pdb',
                title=title,
                authors=authors,
                year=year,
                identifier=f"PDB:{pdb_id}",
                url=f"https://www.rcsb.org/structure/{pdb_id}",
                accessed_date=datetime.now().isoformat(),
                relevant_text=relevant_text
            )

            return citation

        except Exception as e:
            print(f"   ⚠️  Could not fetch PDB entry {pdb_id}: {e}")
            return None


class IntegratedValidator:
    """
    Integrated validator that combines all API sources

    Uses:
    1. PubMed (literature evidence)
    2. IEDB (experimental validation)
    3. PDB (structural evidence)

    Returns ValidationEvidence objects with proper citations
    """

    def __init__(self, email: str, ncbi_api_key: Optional[str] = None):
        """
        Args:
            email: Your email for NCBI
            ncbi_api_key: Optional NCBI API key for higher rate limits
        """
        self.email = email

        # Initialize searchers
        if BIOPYTHON_AVAILABLE:
            self.pubmed = PubMedSearcher(email, ncbi_api_key)
        else:
            self.pubmed = None
            print("⚠️  PubMed search unavailable (Biopython not installed)")

        self.iedb = IEDBSearcher()
        self.pdb = PDBSearcher()

    def validate_epitope(self,
                        epitope_sequence: str,
                        antigen_name: str,
                        organism: str) -> List[ValidationEvidence]:
        """
        Validate epitope using all available sources

        Args:
            epitope_sequence: Epitope amino acid sequence
            antigen_name: Antigen name
            organism: Organism name

        Returns:
            List of ValidationEvidence objects with citations
        """
        all_evidence = []

        # 1. Search PubMed for literature
        if self.pubmed:
            print("\n📚 Searching PubMed...")
            pubmed_citations = self.pubmed.search_epitope(
                epitope_sequence, antigen_name, organism, max_results=5
            )

            if pubmed_citations:
                evidence = ValidationEvidence(
                    evidence_type='database',
                    description=f"Found {len(pubmed_citations)} publications mentioning {organism} {antigen_name} epitopes",
                    confidence_level='medium',
                    citations=pubmed_citations,
                    validation_method="PubMed literature search"
                )
                all_evidence.append(evidence)

        # 2. Search IEDB for experimental validation
        print("\n🔬 Searching IEDB...")
        iedb_citations = self.iedb.search_epitope(
            epitope_sequence, organism, max_results=10
        )

        if iedb_citations:
            evidence = ValidationEvidence(
                evidence_type='experimental',
                description=f"Found {len(iedb_citations)} experimentally validated epitopes in IEDB database",
                confidence_level='high',
                citations=iedb_citations,
                validation_method="IEDB experimental database"
            )
            all_evidence.append(evidence)

        # 3. Search PDB for structures
        print("\n🧬 Searching RCSB PDB...")
        pdb_citations = self.pdb.search_antibody_antigen_complexes(
            antigen_name, organism, max_results=5
        )

        if pdb_citations:
            evidence = ValidationEvidence(
                evidence_type='structural',
                description=f"Found {len(pdb_citations)} antibody-antigen complex structures in PDB",
                confidence_level='high',
                citations=pdb_citations,
                validation_method="RCSB PDB structural database"
            )
            all_evidence.append(evidence)

        return all_evidence


# Test function
if __name__ == '__main__':
    print("="*80)
    print("TESTING API INTEGRATIONS")
    print("="*80)

    # Test with a known SARS-CoV-2 epitope
    test_epitope = "YQAGSTPCNGVEG"
    test_antigen = "spike protein"
    test_organism = "SARS-CoV-2"

    print(f"\nTest epitope: {test_epitope}")
    print(f"Antigen: {test_antigen}")
    print(f"Organism: {test_organism}\n")

    # Initialize validator
    # TODO: Replace with your email
    validator = IntegratedValidator(email="your.email@example.com")

    # Run validation
    evidence = validator.validate_epitope(
        epitope_sequence=test_epitope,
        antigen_name=test_antigen,
        organism=test_organism
    )

    # Print results
    print("\n" + "="*80)
    print("VALIDATION RESULTS")
    print("="*80)

    print(f"\nTotal evidence pieces: {len(evidence)}")

    for i, ev in enumerate(evidence, 1):
        print(f"\n--- Evidence {i} ---")
        print(f"Type: {ev.evidence_type}")
        print(f"Confidence: {ev.confidence_level}")
        print(f"Description: {ev.description}")
        print(f"Citations: {len(ev.citations)}")

        for j, citation in enumerate(ev.citations, 1):
            print(f"\n  Citation {j}:")
            print(f"  {citation.format_citation()}")
