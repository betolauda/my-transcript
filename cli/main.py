#!/usr/bin/env python3
"""
Unified CLI for my-transcript audio transcription pipeline.

Provides modern command-line interface while preserving backward compatibility
with existing scripts: transcribe.py, episode_process.py, and detect_economic_terms_with_embeddings.py
"""

import click
import sys
import os
from pathlib import Path


@click.group()
@click.option('--config', help='Configuration file path')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
@click.option('--output-dir', default='outputs', help='Global output directory')
@click.pass_context
def main(ctx, config, verbose, output_dir):
    """Spanish audio transcription with economic term detection.

    This unified CLI provides access to all pipeline components:
    - transcribe: Convert audio to text using OpenAI Whisper
    - analyze: Process transcriptions with spaCy NLP pipeline
    - detect: Find economic terms using ML embeddings

    All original scripts remain functional for backward compatibility.
    """
    ctx.ensure_object(dict)
    ctx.obj.update({
        'config': config,
        'verbose': verbose,
        'output_dir': output_dir
    })


@main.command()
@click.argument('audio_file', type=click.Path(exists=True))
@click.option('--model', default='base', help='Whisper model size (tiny, base, small, medium, large)')
@click.option('--language', default='Spanish', help='Audio language for transcription')
@click.pass_context
def transcribe(ctx, audio_file, model, language):
    """Transcribe audio file using OpenAI Whisper.

    AUDIO_FILE: Path to the audio file to transcribe (.mp3, .wav, .m4a, etc.)

    Example:
        my-transcript transcribe interview.mp3
        my-transcript transcribe --model small --language Spanish podcast.wav
    """
    _invoke_transcribe_script(audio_file, model, language, ctx.obj)


@main.command()
@click.argument('jsonl_file', type=click.Path(exists=True))
@click.option('--window-size', default=5, help='Co-occurrence window size for graph analysis')
@click.option('--freq-threshold', default=3, help='Minimum frequency threshold for term inclusion')
@click.option('--visualize', is_flag=True, help='Generate interactive HTML visualization (starts local server)')
@click.pass_context
def analyze(ctx, jsonl_file, window_size, freq_threshold, visualize):
    """Analyze transcription with spaCy NLP pipeline.

    JSONL_FILE: Path to the JSONL transcription file from transcribe command

    Creates glossaries and co-occurrence network graphs for economic and cultural terms.

    Example:
        my-transcript analyze outputs/interview_20241201.jsonl
        my-transcript analyze --window-size 7 --freq-threshold 2 transcription.jsonl
        my-transcript analyze --visualize transcription.jsonl
    """
    _invoke_analyze_script(jsonl_file, window_size, freq_threshold, ctx.obj, visualize)


@main.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.option('--similarity-threshold', default=0.7, help='ML similarity threshold for semantic matching')
@click.option('--embeddings/--no-embeddings', default=True, help='Enable/disable SBERT embeddings')
@click.pass_context
def detect(ctx, input_file, similarity_threshold, embeddings):
    """Detect economic terms using SBERT embeddings and FAISS similarity.

    INPUT_FILE: Path to JSONL transcription file to analyze

    Uses advanced ML techniques to find economic indicators and technical terms
    with configurable similarity thresholds.

    Example:
        my-transcript detect outputs/interview_20241201.jsonl
        my-transcript detect --similarity-threshold 0.8 transcription.jsonl
    """
    _invoke_detect_script(input_file, similarity_threshold, embeddings, ctx.obj)


@main.command()
@click.option('--show', is_flag=True, help='Show current configuration')
@click.option('--validate', is_flag=True, help='Validate configuration files')
@click.option('--reset', is_flag=True, help='Reset to default configuration')
def config(show, validate, reset):
    """Manage pipeline configuration settings.

    Example:
        my-transcript config --show
        my-transcript config --validate
    """
    if show:
        _show_config()
    elif validate:
        _validate_config()
    elif reset:
        _reset_config()
    else:
        click.echo("Use --show, --validate, or --reset")


def _invoke_transcribe_script(audio_file, model, language, context):
    """Wrapper to invoke transcribe.py with preserved functionality."""
    try:
        # Add project root to path for importing scripts
        import sys
        from pathlib import Path
        project_root = Path(__file__).parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        # Import the original transcribe module from scripts directory
        import scripts.transcribe as transcribe

        # Temporarily modify the module constants if needed
        original_model = transcribe.WHISPER_MODEL
        original_language = transcribe.LANGUAGE
        original_output_dir = transcribe.OUTPUT_DIR

        try:
            # Override configuration with CLI arguments
            transcribe.WHISPER_MODEL = model
            transcribe.LANGUAGE = language
            if context.get('output_dir'):
                transcribe.OUTPUT_DIR = context['output_dir']

            # Set up sys.argv to match original script expectations
            original_argv = sys.argv.copy()
            sys.argv = ['transcribe.py', audio_file]

            # Load model and transcribe
            whisper_model = transcribe.load_whisper_model(model)
            result = transcribe.transcribe_audio(whisper_model, audio_file, language)
            print(result["text"])

            # Save files with timestamp
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d")
            base_filename = os.path.splitext(os.path.basename(audio_file))[0]
            transcribe.save_transcription_files(result, base_filename, timestamp)

        finally:
            # Restore original values
            transcribe.WHISPER_MODEL = original_model
            transcribe.LANGUAGE = original_language
            transcribe.OUTPUT_DIR = original_output_dir
            sys.argv = original_argv

    except Exception as e:
        click.echo(f"Error during transcription: {e}", err=True)
        sys.exit(1)


def _invoke_analyze_script(jsonl_file, window_size, freq_threshold, context, visualize=False):
    """Wrapper to invoke episode_process.py with preserved functionality."""
    try:
        # Add project root to path for importing scripts
        import sys
        from pathlib import Path
        project_root = Path(__file__).parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        # Import the original episode_process module from scripts directory
        import scripts.episode_process as episode_process

        # Temporarily modify the module constants
        original_window = episode_process.WINDOW_SIZE
        original_threshold = episode_process.FREQUENCY_THRESHOLD

        try:
            # Override configuration with CLI arguments
            episode_process.WINDOW_SIZE = window_size
            episode_process.FREQUENCY_THRESHOLD = freq_threshold

            # Set up sys.argv to match original script expectations
            original_argv = sys.argv.copy()
            sys.argv = ['episode_process.py', jsonl_file]

            # Execute the main processing logic
            # Load spaCy model
            nlp = None
            for model_name in episode_process.SPACY_MODELS:
                try:
                    import spacy
                    nlp = spacy.load(model_name)
                    print(f"Loaded spaCy model: {model_name}")
                    break
                except OSError:
                    continue

            if nlp is None:
                print("Error: No Spanish spaCy model found. Please install one:")
                print("python -m spacy download es_core_news_sm")
                sys.exit(1)

            # Create directories
            for directory in episode_process.OUTPUT_DIRS.values():
                os.makedirs(directory, exist_ok=True)

            # Process the file
            base_filename = os.path.splitext(os.path.basename(jsonl_file))[0]
            economy_keywords, argentinian_terms = episode_process.load_keywords()
            texts = episode_process.load_texts_from_jsonl(jsonl_file)

            if not texts:
                print("Error: No texts found in the input file")
                sys.exit(1)

            # Process with NLP
            term_counter, economy_terms, argentinian_lexicon = episode_process.process_texts_with_nlp(
                texts, nlp, economy_keywords, argentinian_terms
            )

            # Filter and build graph
            common_terms = [term for term, freq in term_counter.items() if freq >= freq_threshold]
            if not common_terms:
                print("Warning: No common terms found. Consider lowering the frequency threshold.")

            G = episode_process.build_cooccurrence_graph(texts, nlp, common_terms)

            # Save results
            if economy_terms:
                episode_process.save_glossary(economy_terms, "economy_glossary", "Economy Glossary")
            if argentinian_lexicon:
                episode_process.save_glossary(argentinian_lexicon, "argentinian_lexicon", "Argentinian Lexicon")

            # Calculate and save graph metrics (always)
            if G.number_of_nodes() > 0:
                import networkx as nx
                degree_centrality = nx.degree_centrality(G)
                betweenness_centrality = nx.betweenness_centrality(G)

                graph_metrics = {
                    "total_nodes": G.number_of_nodes(),
                    "total_edges": G.number_of_edges(),
                    "top_degree_centrality": sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:10],
                    "top_betweenness_centrality": sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
                }
                import json
                with open(f"{episode_process.OUTPUT_DIRS['analysis']}/{base_filename}_graph_metrics.json", "w", encoding="utf8") as f:
                    json.dump(graph_metrics, f, indent=2, ensure_ascii=False)

                # Create visualization only if requested
                if visualize:
                    print("Creating interactive visualization...")
                    episode_process.create_visualization(G, economy_terms, argentinian_lexicon, base_filename)
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
            print(f"Files saved to {episode_process.OUTPUT_DIRS['glossary']}/ and {episode_process.OUTPUT_DIRS['analysis']}/ directories")

        finally:
            # Restore original values
            episode_process.WINDOW_SIZE = original_window
            episode_process.FREQUENCY_THRESHOLD = original_threshold
            sys.argv = original_argv

    except Exception as e:
        click.echo(f"Error during analysis: {e}", err=True)
        sys.exit(1)


def _invoke_detect_script(input_file, similarity_threshold, embeddings, context):
    """Wrapper to invoke detect_economic_terms_with_embeddings.py with preserved functionality."""
    try:
        # Add project root to path for importing scripts
        import sys
        from pathlib import Path
        project_root = Path(__file__).parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        # Set up sys.argv to match original script expectations
        original_argv = sys.argv.copy()
        sys.argv = ['detect_economic_terms_with_embeddings.py', input_file]

        try:
            # Import and call the main function from the detection script in scripts directory
            from scripts.detect_economic_terms_with_embeddings import main as detect_main
            detect_main()

        finally:
            sys.argv = original_argv

    except Exception as e:
        click.echo(f"Error during detection: {e}", err=True)
        sys.exit(1)


def _show_config():
    """Show current configuration settings."""
    try:
        from config.config_loader import get_config
        config = get_config()

        click.echo("Current Configuration:")
        click.echo(f"  Embeddings enabled: {config.use_embeddings}")
        click.echo(f"  Similarity threshold: {config.similarity_threshold}")
        click.echo(f"  Context window: {config.context_window}")
        click.echo(f"  Output directories: {config.get_output_dirs()}")

    except Exception as e:
        click.echo(f"Error loading configuration: {e}", err=True)


def _validate_config():
    """Validate configuration files."""
    try:
        from config.config_loader import get_config
        config = get_config()

        if config.validate():
            click.echo("✓ Configuration is valid")
        else:
            click.echo("✗ Configuration validation failed", err=True)
            sys.exit(1)

    except Exception as e:
        click.echo(f"Error validating configuration: {e}", err=True)
        sys.exit(1)


def _reset_config():
    """Reset configuration to defaults."""
    click.echo("Configuration reset functionality not yet implemented.")
    click.echo("Please manually edit config/settings.json to reset to defaults.")


if __name__ == '__main__':
    main()