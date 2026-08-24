# LLM based TT Visa FAQ Chatbot

A Retrieval-Augmented Generation (RAG) chatbot for Trinidad and Tobago eVisa FAQs, built with LangChain. 

Features multilingual support, streaming responses, and PDF-based document retrieval to answer visa application questions accurately.

## Rationale
The official FAQs for Trinidad and Tobago’s electronic Visa (eVisa) system are currently available only in English, which creates a barrier for non-English speakers applying for entry. This project provides a multilingual, AI-powered chatbot that allows users to access accurate visa information in their own language. This system ensures that applicants can get reliable answers without needing English proficiency.

## Features

- **RAG Architecture**: Combines document retrieval with language model generation for accurate, context-aware responses
- **Multilingual Support**: Built-in translation capabilities for multiple languages
- **Streaming Responses**: Real-time output streaming for better user experience
- **Vector Store Integration**: Uses HuggingFace embeddings for semantic document search
- **Interactive CLI**: Simple command-line interface for chatbot interaction
- **Google Gemini Integration**: Powered by Gemini 2.5 Flash Lite language model

## How It Works

1. **Startup**: The chatbot loads and processes the TT eVisa FAQ PDF (`data/TT_Visa_FAQ.pdf`) at initialization
2. **Document Chunking**: PDF content is split into 1500-character chunks (see [Retrieval Evaluation](#retrieval-evaluation) for why this size was chosen)
3. **Embedding**: Each chunk is embedded using HuggingFace's `all-mpnet-base-v2` model
4. **Query Processing**: User questions are matched against embedded chunks using semantic similarity
5. **Response Generation**: Google Gemini generates concise answers based on the most relevant chunks
6. **Multilingual Support**: Automatic language detection and translation for non-English queries

## Demo
This demo showcases the chatbot accurately answering questions about Trinidad and Tobago eVisa applications based on the official FAQ document, with support for multiple languages.

![Demo Screenshot](images/v2_FAQ_Chatbot_Demo/FAQDemo1.png)


## Project Structure

```
llm-chatbot/
|
├── data/
│   └──TT_Visa_FAQ.pdf           #FAQ document
├── src/
│   └── agent/
│       ├── __init__.py
│       ├── __main__.py          # Entry point
│       ├── cli.py               # Command-line interface
│       ├── agent_builder.py     # Agent construction logic
│       ├── model.py             # LLM initialization
│       ├── config.py            # Configuration settings
│       ├── tools.py             # Agent tools (retrieval, etc.)
│       ├── streaming.py         # Response streaming logic
│       ├── translation.py       # Translation utilities
│       └── vectorstore.py       # Vector database management
├── tests/
│   └── test_agent.py            # Unit tests
├── requirements.txt             # Python dependencies
├── pyproject.toml              # Project metadata
├── .env.example                # Environment variables template
└── README.md
```

## Prerequisites

- Python 3.12+
- Google Gemini API key
- HuggingFace access token 
- LangSmith API key


## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/ChrisDanAsh/RAG-LLM-Chatbot.git
cd RAG-LLM-Chatbot/llm-chatbot
```

### 2. Create Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

Copy the example environment file and add your API keys:

```bash
cp .env.example .env
```

Edit `.env` and add your keys:

```env
GOOGLE_API_KEY=your_google_api_key_here
HF_TOKEN=your_huggingface_token_here
LANGSMITH_API_KEY=your_langsmith_key_here
```

**Getting API Keys:**
- **Google Gemini**: https://ai.google.dev/
- **HuggingFace**: https://huggingface.co/settings/tokens
- **LangSmith**: https://smith.langchain.com/

### 5. Install Package (Editable Mode)

```bash
pip install -e .
```

## Usage

### Run the Chatbot

```bash
python -m agent
```

The CLI will start and prompt you for input. Type your questions and press Enter. The chatbot will stream responses in real-time.

**Example interaction:**
```
You: Who can apply a Visa online?
Assistant: [streaming response with context from vector store] To determine eligibility for an online visa application,you be asked a series of questions durirng the application process.  If you meet the criteria, you can proceed to complete the online form 

You: exit
```

Type `exit`, `quit`, or press `Ctrl+C` to exit.

### Run Tests

The project includes comprehensive unit tests, integration tests, and configuration validation.

Note: Ensure that you are in the llm-chatbot directory to run the tests

#### Run All Tests

```bash
pytest tests/ -v
```

#### Run Specific Test Types
```bash
# Run only integration tests
pytest tests/ -v -m "integration"

# Run with code coverage report
pytest tests/ --cov=agent --cov-report=html

# Run a specific test file
pytest tests/test_agent.py -v

# Run a specific test function
pytest tests/test_agent.py::test_version -v
```

### Code Formatting

The project uses Black and Flake8 for code quality:

```bash
# Format code
black src/ tests/

# Check linting
flake8 src/ tests/
```

## Configuration

Edit `src/agent/config.py` to customize:
- **DOC_PATH**: Path to the FAQ PDF (default: `data/TT_Visa_FAQ.pdf`)
- **Chat Model**: Change the LLM (default: `gemini-2.5-flash-lite`)
- **Embedding Model**: Change the embedding model for vector search
- **Vector Store Settings**: Adjust chunk size (default: `1500`), overlap (default: `0`), retrieval `k` (default: `4`), and storage path

## Retrieval Evaluation

Before settling on the chunk size above, retrieval was evaluated in isolation from the LLM — i.e. "does the vector search find the right FAQ page for a question," independent of whether Gemini writes a good final answer.

**Method**: 20 hand-written questions (`eval/questions.yaml`), each labelled with the correct source page number in the PDF rather than a chunk ID, since chunk IDs change whenever the chunk size changes but page numbers don't. 17 of these questions were in scope and answerable via info in the document and 3 were out of scope to test whether an unanswerable question still gets retrieved with high confidence. Each question's retrieved results were scored with **recall@k** (did the correct page appear in the top-k results?) and **MRR** (how highly was it ranked?) via `eval/run_eval.py`.

**Result**, comparing the original `chunk_size=500` against `chunk_size=1500`:

| Metric | chunk_size=500 | chunk_size=1500 |
|---|---|---|
| recall@1 | 0.76 | 0.76 |
| recall@3 | 0.94 | 0.88 |
| recall@4 | 0.94 | 0.94 |
| recall@6 | 0.94 | 0.94 |
| MRR | 0.84 | 0.84 |

**Takeaway**: chunk_size=1500 has a real, measurable retrieval cost — recall@3 drops from 0.94 to 0.88. One question ("Can I pause my application and finish it another day?", gold page 9) ranks #3 at chunk_size=500 but #4 at chunk_size=1500: bundling page 9's content into one larger chunk dilutes its embedding away from a sharp match on that narrow question, letting three topically-adjacent pages (session timeout, no-changes-after-submit, resuming a saved application) edge it out of the top 3. This is the standard large-chunk trade-off (bigger chunks = more complete context per chunk, but a less focused embedding per chunk) showing up concretely, not sampling noise — it's fully reproducible.

It's also a small, bounded cost: only 1 of 17 labelled in scope questions is affected, and only at k=3 specifically — recall@1 is unchanged, and recall@4/recall@6 are identical between the two configs. Since the app retrieves `k` chunks per query, this trade-off is directly actionable: `RETRIEVAL_K` was raised from 3 to **4**, which exactly recovers it (recall@4 at chunk_size=1500 matches chunk_size=500's ceiling) at the cost of one extra ~150-400 token chunk per query.

Separately, one question ("Am I eligible to apply for an eVisa online?", gold page 1) never appears in the top 6 results at *either* chunk size — a pre-existing retrieval gap unrelated to this change, left unresolved (would need a different embedding model or restructured chunks, e.g. prepending each chunk with its FAQ heading, to fix).

None of this addresses answer *completeness* though — recall@k only checks whether a page's chunk was retrieved at all, not whether that chunk contains the full answer, so it couldn't detect the actual problem that motivated the chunk-size change in the first place. That was found separately (`eval/compare_chunk_sizes.py`): the PDF's median FAQ page is 572 characters and the longest is 1013, so chunk_size=500 was cutting answers roughly in half. A multi-part question ("how many documents can I upload and can I pay by credit card") demonstrated this concretely — at `chunk_size=500` the top results were two disconnected fragments of the same page, both missing the question heading that gives them context; at `chunk_size=1500` the same page returned as one complete, self-contained answer. The chunk size was changed on that structural evidence, with `RETRIEVAL_K` raised to offset the ranking cost it introduced — a reminder that recall@k is the standard retrieval metric, but it's blind to completeness and needs pairing with qualitative spot-checks like this to see the full picture.

To reproduce:
```bash
python eval/run_eval.py --k 1 2 3 4 5 6              # recall@k / MRR comparison, all k values
python eval/compare_chunk_sizes.py                    # side-by-side retrieved chunk text
```

## Known Limitations

- **Free Tier Rate Limit**: Google Gemini free tier has a limit of 20 requests/minute
  - Upgrade to a paid tier for production use: https://ai.google.dev/pricing
  - Rate limit errors will show retry messages
  
- ****PDF Document**: 
  - The chatbot only answers questions based on the TT eVisa FAQ PDF in `data/TT_Visa_FAQ.pdf`. To use a different document, replace the PDF and update `DOC_PATH` in `src/agent/config.py`. 
  
  - This method was selected since the website hosting the document does not allow the chatbot to access it directly.

- **Initialization time**: 
  - **First run**: Downloads the embedding model (~90MB, one-time download) and processes the PDF (~1 minute total)
  - **Subsequent runs**: Loads and processes the PDF at startup (~3-5 seconds)
  - The vectorstore is built in-memory at each startup, not persisted between sessions


## Next Steps
- Implement web application with user friendly UI

## Acknowledgements
- Code based on RAG Tutorial from [https://docs.langchain.com/oss/python/langchain/rag#google-gemini]
- Built with [LangChain](https://github.com/langchain-ai/langchain)
- Powered by [Google Gemini](https://ai.google.dev/)
- Embeddings from [HuggingFace](https://huggingface.co/)

## Contact

- **GitHub**: [@ChrisDanAsh](https://github.com/ChrisDanAsh)
