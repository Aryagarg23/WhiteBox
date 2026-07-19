# WhiteBox

Turns a medical textbook into a graph so an AI's answer can be traced node by node instead of trusted as a black box.

Built in 36-ish hours at The Future of Data (Cincinnati, November 2024). 1st place.

## What it does

WhiteBox takes Davidson's Principles and Practice of Medicine and breaks it into a Neo4j graph of about 60,000 nodes and 130,000 relationships. Ask it a medical question and instead of a model generating an answer from nowhere, it walks a path through that graph and shows you the path.

The bet behind it: in a field like medicine, "trust me" isn't good enough. If a model tells you a symptom points to a diagnosis and a diagnosis points to a treatment, you should be able to see the specific nodes and edges it walked to get there, not just the final sentence.

## How it works

- A question comes in through the `/getNodes` endpoint (`server/Graph_RAG.py`).
- Each word in the question gets embedded with `all-MiniLM-L6-v2` (sentence-transformers) and scored against the whole question by cosine similarity, so filler words drop out and only terms above a 0.25 threshold survive.
- Those terms drive a Cypher query against Neo4j, pulling candidate `Document` nodes that touch them.
- Candidates get re-ranked by cosine similarity between the question and each document's title + text, and the top 3 are kept.
- Their graph neighborhoods get pulled and handed to `microsoft/Phi-3-mini-4k-instruct`, which generates the answer from that neighborhood instead of from general knowledge.
- The same subgraph is returned as nodes and edges and rendered in the browser with vis.js, so the answer and the path it came from sit side by side.
- Model selection and fine-tuning, including a LLaMA 3.1 8B fine-tune for summarization and classification, was Arya's part of the build, feeding the graph and the retrieval step above.
- The graph itself is loaded from a Neo4j dump (`Neo4j/neo4j.dump`, restored by `Neo4j/load-data.sh`); `scripts/DB_labels_fix.py` merges near-duplicate node labels the extraction pass left behind.

## Prototype

The real pipeline needs Neo4j loaded with the full graph and two models resident in memory, not something you casually clone and run. `prototype/graph_qa_demo.py` is a toy stand-in: a 25-node dict graph of symptoms, conditions, and treatments, walked with plain BFS instead of a model call. The point survives the downgrade: the answer *is* the path, and the path is the whole explanation.

No hackathon metrics were logged, so there are no charts of results here — just two explanatory diagrams. `python prototype/graph_qa_demo.py` regenerates both locally (numpy + matplotlib, nothing else).

The first is a real BFS walk on the toy graph, asking "what treats shortness of breath?" and tracing the shortest path to a treatment (the rest of the toy graph shown as hairlines for context):

![Traced path from 'shortness of breath' to 'diuretics', 2 hops through a 25-node toy graph, rest of the graph shown as hairlines. Tagged "toy graph, real walk."](https://vircgxpcwyvniemqmdyi.supabase.co/storage/v1/object/public/media/writing/WhiteBox/traced_path.png)

The second is a concept sketch of the real architecture as built at the hackathon: textbook to graph at build time, question to answer-with-path at query time.

![Architecture diagram: Davidson's textbook chunked and classified by a fine-tuned LLaMA 3.1 8B into a Neo4j graph (~60k nodes / ~130k relationships, real figures from the build), then a clinician's question runs through MiniLM embedding, Cypher retrieval, and Phi-3-mini analysis to produce an answer returned with its graph path. Tagged "architecture as built at the hackathon."](https://vircgxpcwyvniemqmdyi.supabase.co/storage/v1/object/public/media/writing/WhiteBox/pipeline.png)

## Team

- Arya Garg — open-source model selection and fine-tuning. [github.com/Aryagarg23](https://github.com/Aryagarg23)
- [Kaaustaaub Shankar](https://kaaustaaub.netlify.app/) — RAG. [github.com/KaaustaaubShankar](https://github.com/KaaustaaubShankar)
- [Raihan Rafeek](https://www.rai-1975.com/) — database and API.

Originally hacked together in [KaaustaaubShankar/WhiteBox](https://github.com/KaaustaaubShankar/WhiteBox).

## Links

- [Devpost project](https://devpost.com/software/whitebox-zn5u2q)
- [Writeup](https://aryagarg23.com/writing/whitebox)
- [aryagarg23.com](https://aryagarg23.com)
- [Devpost profile](https://devpost.com/Aryagarg23)

## More hackathon builds

- [Gyrus](https://github.com/Aryagarg23/Gyrus) — agentic browser that supports curiosity instead of replacing it (WeaveHacks 2025)
- [G-Code-Assembler](https://github.com/Aryagarg23/G-Code-Assembler) — G-code assembly + STL visualization (MakeUC 2024, Kinetic Vision winner)
- [Terminally-Addicted](https://github.com/Aryagarg23/Terminally-Addicted) — Spotify, GitHub, GPT and YouTube without leaving the terminal (HackOHI/O 2024)
- [Memento](https://github.com/Aryagarg23/Memento) — digital memory journal for Alzheimer's patients and caregivers (RevolutionUC 2024, 3rd overall)
- [Buycott](https://github.com/Aryagarg23/Buycott) — barcode scan -> parent company -> NLP stance on social issues (MakeUC 2023, 1st overall)
- [SignLink](https://github.com/Aryagarg23/SignLink) — video calls with real-time ASL fingerspelling to text (BoilerMake X 2023)
- [Kuka Arm Viz](https://github.com/Aryagarg23/Visualizing-Kuka-7-Node-Robot-Arm) — interactive 7-DOF robot arm in WebGL with inverse kinematics (RevolutionUC 2023)
- [Hi-Five](https://github.com/Aryagarg23/Hi-Five) — anonymous friend-matching on OCEAN personality vectors (SASEhack 2024)
- [Friction](https://github.com/Aryagarg23/Friction) — speculative OS + hardware that protects flow state with physical friction (Fig Build 2026)
