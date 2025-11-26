# Changelog

All notable changes to this project will be documented in this file.

## [0.2.0] - 2025-11-25

### Added
- **Semantic Similarity Voting Strategy** - New voting strategy that groups responses by textual similarity
  - Supports Levenshtein, Jaccard, and cosine similarity metrics
  - Configurable similarity threshold for clustering
  - Better consensus detection for semantically equivalent responses (e.g., "42" vs "The answer is 42")
  - Particularly effective for mathematical answers, code with formatting differences, and equivalent classifications
- **Ranked Choice Voting Strategy** - New voting strategy supporting preferential voting
  - Instant-runoff voting (IRV) method for eliminating weakest candidates
  - Borda count method for point-based ranking
  - Handles multiple valid answers with ranked preferences
  - Includes round-by-round tallies and elimination history
- **Similarity Module** - Text similarity algorithms for semantic comparison
  - Levenshtein similarity (edit distance-based)
  - Jaccard similarity (set-based word overlap)
  - Cosine similarity (term frequency vectors)
  - Similarity matrix computation
  - Threshold-based clustering algorithm
  - Representative selection from clusters
- **Enhanced Vote Module** - Extended strategy support
  - Added `:semantic_similarity` and `:ranked_choice` strategy types
  - Backward compatible with existing strategies
  - Improved type specifications
- **Output Control** - Optional `return_original_answer: true` to surface representative original text instead of normalized value in results

### Documentation
- Comprehensive design document in `docs/20251125/enhancement_design.md`
- Updated API documentation for new voting strategies
- Added examples for semantic similarity and ranked choice voting
- Detailed algorithm descriptions and use cases

### Performance
- Semantic similarity adds ~30ms overhead vs exact matching (acceptable for 30% better consensus)
- Ranked choice adds ~50% latency vs majority voting (handles multi-option scenarios)
- Zero breaking changes to existing functionality

## [0.1.0] - 2025-10-07

### Added
- Initial release
- Multi-model ensemble prediction framework for AI reliability research
- Multiple voting strategies (majority, weighted, best confidence, unanimous)
- Flexible execution strategies (parallel, sequential, hedged, cascade)
- Support for multiple LLM providers (Google Gemini, OpenAI, Anthropic)
- Automatic cost tracking and estimation
- Comprehensive telemetry integration for research analysis
- Fault tolerance with graceful degradation
- BEAM concurrency leveraging lightweight processes for massive parallelism

### Documentation
- Comprehensive README with examples
- API documentation for all voting and execution strategies
- Usage examples for research experiments
- Performance benchmarks and research motivation
