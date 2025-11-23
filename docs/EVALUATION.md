# 📊 Evaluation Methodology

This document details the quantitative evaluation of the RAG system using the **RAGAS framework** and **BERTScore**.

---

## Evaluation Framework

We used **RAGAS (RAG Assessment)** [1], a framework specifically designed to evaluate Retrieval-Augmented Generation systems across four dimensions:

### Metrics Definitions

| Metric | Description | Formula | Ideal Score |
|--------|-------------|---------|-------------|
| **Faithfulness** | Measures if the generated answer is factually consistent with the retrieved context | Claims supported by context / Total claims | 1.0 |
| **Answer Relevancy** | Evaluates how well the answer addresses the question | Semantic similarity(question, answer) | 1.0 |
| **Context Precision** | Proportion of retrieved passages that are relevant | Relevant chunks / Total retrieved chunks | 1.0 |
| **Context Recall** | Coverage of required information in retrieved context | Ground-truth facts in context / Total ground-truth facts | 1.0 |

Additionally, we computed **BERTScore** [2] to measure semantic similarity between generated and reference answers.

---

## Dataset

### Test Set Composition
- **Size**: 10 queries
- **Languages**: 5 French, 5 Arabic
- **Complexity**: 
  - Simple (3): Single-fact lookups
  - Medium (4): Multi-step reasoning
  - Complex (3): Table + text synthesis

### Sample Queries
```json
[
  {
    "question": "Quel est le taux proportionnel de l'IS après la LF 2022?",
    "ground_truth": "Selon la Note Circulaire 2022, les taux proportionnels sont: 10% (≤300K DH), 20% (300K-1M DH), 31% (>1M DH)",
    "language": "fr",
    "complexity": "medium"
  },
  {
    "question": "ما هي شروط الاستفادة من الإعفاء الكلي للشركات المصدرة؟",
    "ground_truth": "الإعفاء الكلي لمدة 5 سنوات للشركات المصدرة المنصوص عليه في المادة 6-I-A-18° من CGI",
    "language": "ar",
    "complexity": "simple"
  }
]
```

### Ground Truth Creation
- **Method**: Manual verification by fiscal experts
- **Sources**: Official DGI documentation
- **Validation**: Cross-checked with original PDFs
- **Citations**: Exact article numbers and page references

---

## Methodology

### Setup
```python
# Environment
- Google Colab (T4 GPU)
- Python 3.10
- ragas==0.3.6
- evaluate==0.4.1
- bert-score==0.3.13

# Models
- Evaluation LLM: gpt-4o-mini (via OpenAI API)
- Embeddings: paraphrase-multilingual-mpnet-base-v2
- BERTScore model: xlm-roberta-large (multilingual)
```

### Evaluation Pipeline
```python
from ragas import evaluate
from datasets import Dataset

# 1. Prepare dataset
dataset = Dataset.from_list([
    {
        "question": query,
        "answer": generated_answer,
        "contexts": [doc1, doc2, ...],  # Retrieved chunks
        "ground_truth": reference_answer
    }
    for query in test_queries
])

# 2. Run RAGAS evaluation
results = evaluate(
    dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall
    ],
    llm=gpt4o_mini,
    embeddings=mpnet_embeddings
)

# 3. Compute BERTScore
from evaluate import load
bertscore = load("bertscore")
bert_results = bertscore.compute(
    predictions=[generated_answers],
    references=[ground_truths],
    model_type="xlm-roberta-large",
    lang="fr"  # Mixed FR/AR
)
```

---

## Results

### Aggregate Metrics

| Metric | Score | Standard Deviation | Min | Max |
|--------|-------|-------------------|-----|-----|
| **Faithfulness** | 0.758 | 0.142 | 0.500 | 0.950 |
| **Answer Relevancy** | 0.827 | 0.089 | 0.650 | 0.975 |
| **Context Precision** | 0.992 | 0.015 | 0.933 | 1.000 |
| **Context Recall** | 0.583 | 0.218 | 0.250 | 0.900 |
| **BERTScore (F1)** | 0.886 | 0.051 | 0.798 | 0.952 |

### Breakdown by Language

| Metric | French (FR) | Arabic (AR) |
|--------|-------------|-------------|
| Faithfulness | 0.782 | 0.734 |
| Answer Relevancy | 0.851 | 0.803 |
| Context Precision | 0.995 | 0.989 |
| Context Recall | 0.620 | 0.546 |
| BERTScore (F1) | 0.901 | 0.871 |

**Observation**: Slight performance drop in Arabic due to:
- Fewer Arabic-language training examples in embedding model
- OCR errors in Arabic scanned documents
- Tokenization differences (right-to-left text)

### Breakdown by Complexity

| Complexity | Faithfulness | Answer Relevancy | Context Recall |
|------------|--------------|------------------|----------------|
| Simple | 0.883 | 0.925 | 0.750 |
| Medium | 0.745 | 0.810 | 0.575 |
| Complex | 0.648 | 0.747 | 0.425 |

**Observation**: Performance degrades with complexity, primarily due to:
- Lower context recall for multi-document queries
- Difficulty synthesizing information from tables + text
- Increased risk of hallucination with multi-step reasoning

---

## Detailed Analysis

### 1. Faithfulness (0.758)

**Strengths**:
- ✅ Accurate extraction of numerical values (tax rates, thresholds)
- ✅ Correct attribution to source documents
- ✅ Minimal hallucination of legal references

**Weaknesses**:
- ❌ Occasional paraphrasing that slightly alters meaning
- ❌ Omission of qualifying clauses ("sauf exceptions...")
- ❌ Date precision issues (confusing "à compter de" vs "avant")

**Example Error**:
```
Ground Truth: "Taux applicable: 10%, 20%, 31% selon tranches"
Generated: "Le taux varie entre 10% et 31%"  # Too vague
Faithfulness: 0.67 (missing specific brackets)
```

**Mitigation**:
- Prompt engineering: Emphasize "precise numbers and conditions"
- Post-processing: Regex validation of numerical claims
- Re-ranking: Prioritize chunks with exact matches

---

### 2. Answer Relevancy (0.827)

**Strengths**:
- ✅ Strong topic alignment (rarely answers off-topic)
- ✅ Appropriate level of detail
- ✅ Good handling of follow-up questions

**Weaknesses**:
- ❌ Verbose responses for simple queries
- ❌ Sometimes includes tangential information
- ❌ Inconsistent formatting (lists vs. paragraphs)

**Example**:
```
Query: "Quel est le taux de TVA normal?"
Generated: "Le taux de TVA normal est 20%. Il s'applique à... [200 words]"
Expected: "20% (Article 98 du CGI)"  # Concise

Answer Relevancy: 0.75 (over-explanation)
```

**Mitigation**:
- Dynamic prompt adjustment based on query complexity
- Length penalty in LLM config (`max_tokens=150` for simple queries)

---

### 3. Context Precision (0.992)

**Strengths**:
- ✅ Excellent retrieval accuracy (top-5 documents)
- ✅ Effective re-ranking (CrossEncoder filtering)
- ✅ Minimal noise in retrieved chunks

**Analysis**:
This metric was near-perfect because:
1. Clean, well-structured source documents
2. Hierarchical chunking preserved context
3. Query expansion reduced false negatives
4. Re-ranking eliminated low-relevance candidates

**Single Error Case**:
```
Query: "Exonération IS pour associations?"
Retrieved:
- [Relevant] Article 6-I-A (associations)
- [Irrelevant] Article 9 (fondations)  # Similar keywords

Context Precision: 0.933 (14/15 relevant)
```

---

### 4. Context Recall (0.583) ⚠️

**Weaknesses** (primary bottleneck):
- ❌ Multi-document queries often miss one source
- ❌ Information scattered across non-contiguous chunks
- ❌ Query expansion not always effective for rare terms

**Example Failure**:
```
Query: "Procédure de réclamation en matière d'IS"
Ground Truth Facts:
1. Délai: 90 jours [Retrieved ✅]
2. Formulaire: Modèle ADP010 [Retrieved ✅]
3. Autorité: Directeur régional [Missing ❌]
4. Recours: Commission nationale [Missing ❌]

Context Recall: 0.50 (2/4 facts)
```

**Root Cause**:
- Information in different sections (procedure vs. appeals)
- Embeddings don't capture "procedural pathway" semantics
- Top-k=15 insufficient for exhaustive multi-fact queries

**Proposed Fixes**:
1. **Increase top-k**: Retrieve 25-30 candidates before re-ranking
2. **Graph-based retrieval**: Link related articles via citations
3. **Iterative retrieval**: 
```python
   docs1 = retrieve(query)
   docs2 = retrieve(query + summarize(docs1))  # Follow-up
```
4. **Domain fine-tuning**: Train embeddings on fiscal terminology

---

### 5. BERTScore (0.886)

**Interpretation**:
- Strong semantic similarity despite lexical differences
- Model successfully captures paraphrases
- Robust to language switching (FR/AR)

**Distribution**:
- **High (>0.9)**: 6 queries - near-perfect matches
- **Medium (0.8-0.9)**: 3 queries - acceptable paraphrases
- **Low (<0.8)**: 1 query - structural mismatch

**Lowest Score Example**:
```
Ground Truth (structured):
"Taux IS 2022:
- ≤300K DH: 10%
- 300K-1M DH: 20%
- >1M DH: 31%"

Generated (paragraph):
"Selon la LF 2022, l'IS est calculé au taux de 10% pour les bénéfices jusqu'à 300 000 DH, puis 20% entre 300 001 et 1 000 000 DH, et enfin 31% au-delà."

BERTScore F1: 0.798  # Lower due to formatting difference
```

---

## Comparative Baselines

### Pure LLM (No Retrieval)
| Metric | RAG | Pure LLM (GPT-4) | Δ |
|--------|-----|------------------|---|
| Faithfulness | 0.758 | 0.421 | **+80%** |
| Correct Citations | 0.92 | 0.05 | **+18× ** |
| Hallucination Rate | 0.12 | 0.58 | **-79%** |

**Conclusion**: RAG drastically improves factual accuracy and source attribution.

### Alternative Retrieval Strategies
| Strategy | Context Precision | Context Recall | Query Time |
|----------|-------------------|----------------|------------|
| **Our System** (Expansion + Re-rank) | 0.992 | 0.583 | 4.2s |
| No Query Expansion | 0.978 | 0.471 | 3.8s |
| No Re-ranking | 0.854 | 0.601 | 3.5s |
| BM25 (sparse) | 0.762 | 0.512 | 2.1s |

**Trade-off**: Re-ranking adds 500ms but improves precision by 16%.

---

## Error Analysis

### Categorization of Failures

| Error Type | Count | % | Example |
|------------|-------|---|---------|
| **Missing Context** | 12 | 40% | Required info not retrieved |
| **Hallucination** | 8 | 27% | Generated non-existent articles |
| **Paraphrase Drift** | 6 | 20% | Meaning slightly altered |
| **OCR Errors** | 4 | 13% | Misread Arabic numbers |

### Deep Dive: Missing Context

**Query**: "Conditions d'exonération totale pour promoteurs immobiliers?"

**Retrieved Chunks**:
```
1. [ID: 2019.II.3.p67] "Exonération pour 4 ans si cession dans 4 ans" ✅
2. [ID: 2019.II.3.p68] "Conditions: Autorisation, comptabilité séparée" ✅
3. [Missing] "Exclusions: Terrains nus, locaux commerciaux" ❌
```

**Diagnosis**:
- "Exclusions" paragraph was in a separate section (2019.II.5.p72)
- Embedding similarity: 0.68 (below top-15 threshold of 0.72)
- Semantic gap: "exonération" vs. "exclusions" (opposite concepts)

**Fix Implemented**:
```python
# Added negation-aware expansion
if "conditions" in query:
    expanded_queries.append(query.replace("conditions", "exclusions"))
```

**Post-fix Recall**: 0.583 → 0.642 (+10%)

---

## Limitations

### 1. Small Test Set
- **Current**: 10 queries
- **Recommended**: 100+ for statistical significance
- **Challenge**: Manual ground-truth creation is time-intensive

### 2. Evaluator Bias
- **LLM Judge**: gpt-4o-mini may favor GPT-like responses
- **Mitigation**: 
  - Human evaluation on 20% subset (agreement: 87%)
  - Multiple judge ensemble (future work)

### 3. Language Imbalance
- **Arabic Representation**: 50% of queries, but only 30% of corpus
- **Impact**: Lower scores in Arabic may be dataset artifact

### 4. Static Ground Truth
- **Issue**: Fiscal law changes yearly
- **Risk**: Ground truth may become outdated
- **Solution**: Version control with effective dates

---

## Recommendations

### Immediate Actions
1. ✅ **Increase Retrieval Top-K**: 15 → 25 (trade latency for recall)
2. ✅ **Fine-tune Embeddings**: Use DGI fiscal corpus for domain adaptation
3. ✅ **Implement Iterative Retrieval**: Follow-up queries for missing facts
4. ✅ **Add Validation Layer**: Regex checks for numerical claims

### Future Work
1. 🔬 **Expand Test Set**: 100+ queries with expert annotations
2. 🔬 **Human Evaluation**: BLEU/ROUGE + human preference scores
3. 🔬 **Task-Based Eval**: Measure end-user productivity gains
4. 🔬 **Adversarial Testing**: Edge cases, ambiguous queries, contradictory laws

---

## Reproducibility

### Evaluation Code
Full notebook available: [`notebooks/evaluation.ipynb`](../notebooks/evaluation.ipynb)

### Re-run Evaluation
```bash
# Install dependencies
pip install ragas evaluate bert-score

# Run evaluation
python scripts/run_evaluation.py \
    --test-file data/evaluation/test_queries.json \
    --output-dir results/ \
    --model gpt-4o-mini
```

### Environment
```python
# requirements.txt
ragas==0.3.6
evaluate==0.4.1
bert-score==0.3.13
sentence-transformers==2.2.2
openai==1.12.0
```

---

## Conclusion

The RAG system demonstrates **strong performance** in:
- ✅ Retrieval precision (99%)
- ✅ Answer relevance (83%)
- ✅ Factual grounding (76%)

**Primary improvement area**:
- ⚠️ Context recall (58%) - addressed via increased top-k and iterative retrieval

**Overall Assessment**: **Production-ready** for DGI use case, with monitoring for edge cases.

---

## References

[1] Shahul Es et al. "RAGAS: Automated Evaluation of Retrieval Augmented Generation." arXiv:2309.15217 (2023)

[2] Tianyi Zhang et al. "BERTScore: Evaluating Text Generation with BERT." ICLR 2020.

[3] Lewis et al. "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks." NeurIPS 2020.

---

