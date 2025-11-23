# src/config.py
import os
from pathlib import Path

# Project root directory
ROOT_DIR = Path(__file__).parent.parent

# --- Data and Database Paths ---
# Directory containing the raw JSON chunk files
DATA_DIR = ROOT_DIR / "data"

# Path where the persistent ChromaDB vector database will be stored
DB_PATH = str(ROOT_DIR / "vector_db")

# --- Indexing and Model Configuration ---
# Name of the ChromaDB collection
COLLECTION_NAME = "moroccan_fiscal_law"

# Sentence Transformer model for creating multilingual embeddings
# 'paraphrase-multilingual-mpnet-base-v2' is a strong choice for FR/AR/EN
EMBEDDING_MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"

# Batch size for indexing documents into ChromaDB
INDEXING_BATCH_SIZE = 64

# --- LLM and Prompt Configuration ---

# Dictionary of LLM Models available via OpenRouter
MODEL_PAIRS = {
    "Duel DeepSeek": {
        "name": "DeepSeek (Classique vs Raisonnement)",
        "classic": {"name": "DeepSeek V3 (Classique)", "id": "deepseek/deepseek-chat-v3-0324"},
        "reasoning": {"name": "DeepSeek R1 (Raisonnement)", "id": "deepseek/deepseek-r1"}
    },
    "Duel Mistral": {
        "name": "Mistral (Puissance vs Raisonnement)",
        "classic": {"name": "Mixtral 8x22B (Puissance)", "id": "mistralai/mixtral-8x22b-instruct"},
        "reasoning": {"name": "Magistral Medium", "id": "mistralai/magistral-medium-2506"}
    },
    "Duel Llama": {
        "name": "Llama (Test de l'Échelle)",
        "classic": {"name": "Llama 3.1 70B (Puissant)", "id": "meta-llama/llama-3.1-70b-instruct"},
        "reasoning": {"name": "Llama 3.1 405B (Très Puissant)", "id": "meta-llama/llama-3.1-405b-instruct"}
    },
    "Duel Kimi": {
        "name": "Kimi (Puissance vs Agentivité)",
        "classic": {"name": "Kimi Dev 72B", "id": "moonshotai/kimi-dev-72b:free"},
        "reasoning": {"name": "Kimi K2 (Agent)", "id": "moonshotai/kimi-k2"}
    },
    "Duel Qwen": {
        "name": "Qwen (Puissance Dense vs Raisonnement Spécialisé)",
        "classic": {"name": "Qwen2.5 72B Instruct", "id": "qwen/qwen-2.5-72b-instruct"},
        "reasoning": {"name": "QwQ 32B (Raisonnement)", "id": "qwen/qwq-32b"}
    }
}
LOCAL_MODELS = {
    "mistral": {
        "name": "Mistral 7B Instruct",
        "id": "mistral"
    },
    "llama3": {
        "name": "Llama 3 8B Instruct",
        "id": "llama3"
    },
    "gemma": {
        "name": "Gemma 7B Instruct",
        "id": "gemma"
    }
}

# Adaptive system prompts based on language
SYSTEM_PROMPT_FR = """
Vous êtes un expert en droit fiscal marocain, spécialisé dans les documents fiscaux comme la Note Circulaire. Votre mission est de fournir des réponses précises et exhaustives basées EXCLUSIVEMENT sur les extraits fournis. Répondez en français.

RÈGLES STRICTES :
1.  **Source unique** : Basez votre réponse UNIQUEMENT sur les informations des documents fournis.
2.  **Citations obligatoires** : Chaque information doit être suivie de sa source au format (Source: Note Circulaire [année], ID: [chunk_id]).
3.  **Information manquante** : Si l'information n'est pas dans le contexte, répondez clairement : "Cette information n'est pas présente dans les extraits fournis."
"""

SYSTEM_PROMPT_AR = """
أنت خبير في القانون الضريبي المغربي، متخصص في الوثائق الضريبية مثل المذكرة الدورية. مهمتك هي تقديم إجابات دقيقة وشاملة بناءً على المقتطفات المقدمة حصريًا. أجب باللغة العربية.

قواعد صارمة:
1.  **المصدر الوحيد**: يجب أن تستند إجابتك حصريًا إلى المعلومات الموجودة في المستندات المقدمة.
2.  **الاقتباسات الإلزامية**: يجب أن تتبع كل معلومة مصدرها بالتنسيق التالي (المصدر: مذكرة دورية [السنة]، المعرف: [chunk_id]).
3.  **المعلومات المفقودة**: إذا لم تكن المعلومات موجودة في السياق، أجب بوضوح: "هذه المعلومات غير موجودة في المقتطفات المقدمة."
"""