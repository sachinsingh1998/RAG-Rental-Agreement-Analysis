# RAG pipeline (for analyzing rental agreement documents)

This is a small RAG setup for rental agreement PDFs. It cuts each PDF text into chunks, stores those  in Chroma Database as embeddings / vector, and later answers a question using only the pieces that match that question.

The model does not read the whole PDF at question time. First we cut the document into pieces and turn each piece into a list of numbers (an embedding). When a question comes in, we pick the 5 pieces closest to that question and send only those to the model. This approach reduces the amount of LLM calls and thus the tokens.

## What is in this folder

```
documents/     Put the rent agreement PDFs here. Ingestion reads this folder.
chroma_db/     Local Chroma database. Created when you run ingestion.
chunking.py    Read a PDF and cut it into chunks.
helper.py      OpenAI embedding calls.
ingestion.py   Read every PDF, embed the chunks, save them in Chroma.
retrieval.py   Ask a question, fetch chunks, print the answer.
main.py        A small chunking check on documents/.
req.txt        Packages to install.
.env           API key and model name. Do not commit this.
```

## Setup

You need Python 3. Make a virtual environment so the packages stay inside this project.

```bash
python -m venv myvenv
```

On Windows:

```bash
myvenv\Scripts\activate
pip install -r req.txt
```

On Mac or Linux, activate with `source myvenv/bin/activate`, then run the same pip command.

Create a `.env` file in the project root:

```
OPENAI_API_KEY=your_key_here
MODEL_NAME=gpt-4.1-mini
```

The OpenAI client reads `OPENAI_API_KEY` on its own. `MODEL_NAME` is only for the final answer in retrieval. The embedding model is fixed in the code: `text-embedding-3-small`.


## How to run it

1. Put the PDFs in `documents/`. Keep the names simpl
2. Build the index first:

```bash
python ingestion.py
```

This reads every file in `documents/`, makes chunks, gets embeddings, and saves them in the `rental_agreements` collection inside `./chroma_db`.

3. Then ask questions:

```bash
python retrieval.py
```

It asks for two things.

- The file name, with the extension. Type `filename.pdf`, not just `filename`.
- Your question.

Type `exit` at the document prompt when you want to stop.

`main.py` is not the full pipeline. It only chunks `documents/RK.pdf`, and the print lines are commented out. Uncomment them if you want to see the chunks.

## What each part does

### Chunking (`chunking.py`)

`pypdf` reads every page and joins the text into one long string.

That string is then cut like this:

- chunk size: 500 characters
- overlap: 100 characters

The overlap is there so a sentence sitting on the cut does not get lost. The next chunk starts 100 characters before the previous chunk ends.

Each chunk is saved with three things:

- id: `documents/filename.pdf_1`, `documents/filename.pdf_2`, and so on
- text: that 500-character piece
- metadata: `doc_name` (the full path, like `documents/filename.pdf`) and `version` (always `1` for now)

### Embeddings (`helper.py`)

OpenAI model `text-embedding-3-small`, with 300 dimensions.

- Ingestion sends all chunks of one file together, in one batch.
- Retrieval asks for one embedding, for the question only.

Both places must use the same model and the same 300 dimensions. If you change one, change the other too. Otherwise Chroma cannot compare the question with the stored chunks.

### Store (`ingestion.py`)

Chroma lives on disk at `./chroma_db`. The collection name is `rental_agreements`.

The loop is straight: each file in the folder, then chunks, then embeddings, then `collection.add(...)`.

One thing to watch. If you ingest the same file again, the ids are repeated and Chroma throws an error on add. To load fresh data, delete the `chroma_db` folder and run ingestion again, or delete the old ids first.

### Retrieval (`retrieval.py`)

The question is turned into an embedding. Chroma returns the top 5 chunks, but only from the document you named.

The filter is `doc_name == documents/<the name you typed>`.

So the name has to match exactly. For `Ana.pdf`, the stored value is `documents/Ana.pdf`. An extra space, the wrong case, or a missing `.pdf` means no chunks come back.

The terminal prints the id of each match and the first 220 characters, so you can see what context the model got.

Those same 5 chunks go into the prompt. The prompt tells the model to answer like a rental-agreement lawyer in Sydney, and to say "I don't know" when the context does not have the answer.

The answer comes from `client.responses.create`, using `MODEL_NAME` from `.env`.

## What to keep out of git

Ignore `myvenv/`, `.env`, and `chroma_db/`. The virtual environment and the database are local and can be created again. `.env` has your key.

Ignore `documents/` only when the PDFs are personal or very large. Sample PDFs for the lecture can stay in the repo.
