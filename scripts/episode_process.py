import spacy
import json
import sys
import os
from collections import Counter
import networkx as nx
from itertools import combinations
from pyvis.network import Network

# Configuration constants
WINDOW_SIZE = 5
FREQUENCY_THRESHOLD = 3
OUTPUT_DIRS = {"glossary": "glossary", "analysis": "outputs"}
SPACY_MODELS = ["es_core_news_trf", "es_core_news_md", "es_core_news_sm"]


def load_keywords():
    """Return economy and argentinian keyword sets."""
    economy_keywords = {
        'inflación', 'devaluación', 'peso', 'dólar', 'banco', 'central', 'bcra',
        'déficit', 'superávit', 'balanza', 'comercial', 'inversión', 'fmi',
        'mercado', 'capitales', 'exportación', 'importación', 'aranceles',
        'pib', 'gdp', 'crecimiento', 'económico', 'recesión', 'crisis',
        'política', 'monetaria', 'fiscal', 'tasa', 'interés', 'bonos',
        'acciones', 'bolsa', 'merval', 'riesgo', 'país', 'deuda', 'externa',
        'cepo', 'cambiario', 'blue', 'oficial', 'brecha', 'cambiaria'
    }

    argentinian_terms = {
        'che', 'boludo', 'quilombo', 'laburo', 'guita', 'mango', 'pibe', 'piba',
        'bondi', 'colectivo', 'subte', 'porteño', 'bonaerense', 'porteña',
        'asado', 'choripán', 'empanada', 'mate', 'yerba', 'dulce', 'leche',
        'tango', 'boca', 'river', 'maradona', 'messi', 'evita', 'perón',
        'kirchner', 'macri', 'milei', 'cfk', 'cristina', 'mauricio', 'javier'
    }

    return economy_keywords, argentinian_terms


def load_texts_from_jsonl(file_path):
    """Load text data from JSONL file."""
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found")
        sys.exit(1)

    texts = []
    try:
        with open(file_path, encoding="utf8") as f:
            for line in f:
                record = json.loads(line)
                texts.append(record["text"])
        return texts
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)


def process_texts_with_nlp(texts, nlp, economy_keywords, argentinian_terms):
    """Extract terms and return counters for all terms, economy terms, and argentinian terms."""
    term_counter = Counter()
    economy_terms = Counter()
    argentinian_lexicon = Counter()

    for doc in nlp.pipe(texts, batch_size=16):
        for chunk in doc.noun_chunks:
            term = chunk.text.lower().strip()
            if len(term) > 2:
                term_counter[term] += 1

                # Check if it's an economic term
                if any(keyword in term for keyword in economy_keywords):
                    economy_terms[term] += 1

        for ent in doc.ents:
            term = ent.text.lower().strip()
            if len(term) > 2:
                term_counter[term] += 1

                # Check if it's an economic entity
                if any(keyword in term for keyword in economy_keywords):
                    economy_terms[term] += 1

        # Extract Argentinian terms from tokens
        for token in doc:
            if not token.is_stop and token.is_alpha:
                term = token.text.lower()
                if term in argentinian_terms:
                    argentinian_lexicon[term] += 1

    return term_counter, economy_terms, argentinian_lexicon


def build_cooccurrence_graph(texts, nlp, common_terms):
    """Build and return NetworkX graph with co-occurrence relationships."""
    G = nx.Graph()

    for doc in nlp.pipe(texts, batch_size=16):
        tokens = [t.text.lower() for t in doc if not t.is_stop and t.is_alpha and len(t.text) > 2]
        for i in range(len(tokens) - WINDOW_SIZE + 1):
            window = tokens[i : i + WINDOW_SIZE]
            # Only consider terms that appear frequently
            relevant_window = [term for term in window if term in common_terms]
            for a, b in combinations(set(relevant_window), 2):
                if G.has_edge(a, b):
                    G[a][b]["weight"] += 1
                else:
                    G.add_edge(a, b, weight=1)

    return G


def save_glossary(terms_counter, filename_base, title):
    """Save glossary in both JSON and markdown formats."""
    # Prepare data
    glossary_data = {
        "terms": dict(terms_counter.most_common()),
        "total_terms": len(terms_counter),
        "most_frequent": terms_counter.most_common(10)
    }

    # Save JSON
    json_path = f"{OUTPUT_DIRS['glossary']}/{filename_base}.json"
    with open(json_path, "w", encoding="utf8") as f:
        json.dump(glossary_data, f, indent=2, ensure_ascii=False)

    # Save Markdown
    md_path = f"{OUTPUT_DIRS['glossary']}/{filename_base}.md"
    with open(md_path, "w", encoding="utf8") as f:
        f.write(f"# {title}\n\n")
        f.write(f"**Total terms found:** {len(terms_counter)}\n\n")
        f.write("## Most Frequent Terms\n\n")
        f.write("| Term | Frequency |\n")
        f.write("|------|----------|\n")
        for term, freq in terms_counter.most_common(20):
            f.write(f"| {term} | {freq} |\n")
        f.write("\n## All Terms\n\n")
        for term, freq in terms_counter.most_common():
            f.write(f"- **{term}**: {freq}\n")


def create_visualization(G, economy_terms, argentinian_lexicon, base_filename):
    """Generate PyVis network visualization."""
    # Calculate graph metrics
    degree_centrality = nx.degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G)

    # Save graph metrics
    graph_metrics = {
        "total_nodes": G.number_of_nodes(),
        "total_edges": G.number_of_edges(),
        "top_degree_centrality": sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:10],
        "top_betweenness_centrality": sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
    }
    with open(f"{OUTPUT_DIRS['analysis']}/{base_filename}_graph_metrics.json", "w", encoding="utf8") as f:
        json.dump(graph_metrics, f, indent=2, ensure_ascii=False)

    # Create PyVis network with CDN resources to avoid local lib/ dependency
    net = Network(height="750px", width="100%", bgcolor="#ffffff", font_color="black", cdn_resources='remote')
    net.set_options("""
    var options = {
      "physics": {
        "enabled": true,
        "stabilization": {"iterations": 100}
      }
    }
    """)

    # First, add all nodes that will be included
    nodes_to_include = set()
    for u, v, data in G.edges(data=True):
        if data["weight"] >= FREQUENCY_THRESHOLD:
            nodes_to_include.add(u)
            nodes_to_include.add(v)

    # Add nodes with centrality-based sizing
    for node in nodes_to_include:
        size = degree_centrality.get(node, 0) * 50 + 10
        color = "#ff6b6b" if node in economy_terms else "#4ecdc4" if node in argentinian_lexicon else "#95e1d3"
        net.add_node(node, title=f"{node}\nDegree: {degree_centrality.get(node, 0):.3f}",
                    size=size, color=color)

    # Then add edges between existing nodes
    for u, v, data in G.edges(data=True):
        if data["weight"] >= FREQUENCY_THRESHOLD and u in nodes_to_include and v in nodes_to_include:
            net.add_edge(u, v, value=data["weight"], title=f"Co-occurrences: {data['weight']}")

    net.write_html(f"{OUTPUT_DIRS['analysis']}/{base_filename}_graph.html")
    return graph_metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process transcription and extract economic/argentinian terms")
    parser.add_argument("input_file", help="Input JSONL file containing transcription")
    parser.add_argument("--visualize", action="store_true", help="Generate interactive HTML visualization (starts local server)")

    args = parser.parse_args()
    input_file = args.input_file

    # Load the Spanish language model with fallback options
    nlp = None
    for model_name in SPACY_MODELS:
        try:
            nlp = spacy.load(model_name)
            print(f"Loaded spaCy model: {model_name}")
            break
        except OSError:
            continue

    if nlp is None:
        print("Error: No Spanish spaCy model found. Please install one:")
        print("python -m spacy download es_core_news_sm")
        sys.exit(1)

    # Create directories if they don't exist
    for directory in OUTPUT_DIRS.values():
        os.makedirs(directory, exist_ok=True)

    # Extract base filename for output files
    base_filename = os.path.splitext(os.path.basename(input_file))[0]

    # Load configuration and data
    economy_keywords, argentinian_terms = load_keywords()
    texts = load_texts_from_jsonl(input_file)

    if not texts:
        print("Error: No texts found in the input file")
        sys.exit(1)

    # Process texts with NLP
    term_counter, economy_terms, argentinian_lexicon = process_texts_with_nlp(
        texts, nlp, economy_keywords, argentinian_terms
    )

    # Filter common terms by frequency threshold
    common_terms = [term for term, freq in term_counter.items() if freq >= FREQUENCY_THRESHOLD]

    if not common_terms:
        print("Warning: No common terms found. Consider lowering the frequency threshold.")

    # Build co-occurrence graph
    G = build_cooccurrence_graph(texts, nlp, common_terms)

    # Save glossaries
    if economy_terms:
        save_glossary(economy_terms, "economy_glossary", "Economy Glossary")
    if argentinian_lexicon:
        save_glossary(argentinian_lexicon, "argentinian_lexicon", "Argentinian Lexicon")

    # Calculate and save graph metrics (always)
    if G.number_of_nodes() > 0:
        degree_centrality = nx.degree_centrality(G)
        betweenness_centrality = nx.betweenness_centrality(G)

        graph_metrics = {
            "total_nodes": G.number_of_nodes(),
            "total_edges": G.number_of_edges(),
            "top_degree_centrality": sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:10],
            "top_betweenness_centrality": sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
        }
        with open(f"{OUTPUT_DIRS['analysis']}/{base_filename}_graph_metrics.json", "w", encoding="utf8") as f:
            json.dump(graph_metrics, f, indent=2, ensure_ascii=False)

        # Create visualization only if requested
        if args.visualize:
            print("Creating interactive visualization...")
            create_visualization(G, economy_terms, argentinian_lexicon, base_filename)
        else:
            print("Skipping visualization (use --visualize to create interactive graph)")
    else:
        print("Warning: No graph nodes found. Skipping visualization.")
        graph_metrics = {"total_nodes": 0, "total_edges": 0}

    # Print summary
    print(f"Analysis complete!")
    print(f"Economy terms found: {len(economy_terms)}")
    print(f"Argentinian terms found: {len(argentinian_lexicon)}")
    print(f"Graph nodes: {graph_metrics['total_nodes']}")
    print(f"Graph edges: {graph_metrics['total_edges']}")
    print(f"Files saved to {OUTPUT_DIRS['glossary']}/ and {OUTPUT_DIRS['analysis']}/ directories")
