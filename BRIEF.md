# BRIEF — WhiteBox (repo: Aryagarg23/WhiteBox)
Slug: whitebox. Hackathon: The Future of Data (Cincinnati), November 2024. Prizes: 1st place.
Team: Arya Garg (open-source model selection + fine-tuning), Kaaustaaub Shankar (RAG), Raihan Rafeek (database + API). Original team repo: https://github.com/KaaustaaubShankar/WhiteBox
What: hybrid/Graph RAG for explainability — turned Davidson's Principles and Practice of Medicine into a Neo4j graph (~60k nodes, ~130k relationships); answers trace the reasoning path node by node instead of a black-box blob. LLaMA 3.1 8B fine-tune for summarization/classification, MiniLM for Cypher retrieval, Phi-3 3.8B instruct for deeper analysis, Flask + G6-AntV visualization.
Devpost: https://devpost.com/software/whitebox-zn5u2q
Prototype idea: toy knowledge-graph QA — ~25-node dict-based medical mini-graph, answer a query by BFS path-walk, chart 1: the traced subgraph path drawn with matplotlib (nodes+edges, path highlighted in blue, rest hairline), chart 2: traceability — steps you can audit: graph path vs raw LLM (n auditable steps bar).
