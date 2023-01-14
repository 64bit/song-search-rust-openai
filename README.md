# song-search-rust-openai

# Overview

Code and Data for https://gigapotential.dev/blog/song-search-in-rust-using-openai

![Song-Search](screen-recording.svg)

# Getting started

There are two options: basically one Docker image contains data for embedding already, so you can jump right into querying.

- Option 1: docker-compose.data.yaml : The referenced image is built from `ankane/pgvector:v0.4.0` and contains embedding data for all Songs.
- Option 2: docker-compose.yaml: This starts a container with volume mount with no data.

## Option 1: Straight to querying

```
docker compose -f docker-compose.data.yaml up -d
cd song-search
export OPENAI_API_KEY='sk-...'
cargo run --bin query-song
```

This will open query prompt, to exit type ctrl+c


## Option 2: Get Embedding and then query

This step is the most time consuming and incurs approx. $1.5 cost.

```
docker compose -f docker-compose.yaml up -d
cd song-search

#For all songs in data/input dir, fetch embedding from OpenAI and save to DB.
export OPENAI_API_KEY='sk-...'
cargo run --bin create-song-embeddings

```

Thats it, ready for querying:

```
cd song-search
export OPENAI_API_KEY='sk-...'
cargo run --bin query-song
```

This will open query prompt, to exit type ctrl+c
