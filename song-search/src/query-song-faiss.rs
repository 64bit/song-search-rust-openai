use std::io::Write;

use anyhow::{anyhow, Result};
use async_openai::{types::CreateEmbeddingRequestArgs, Client};
use hora::{
    core::ann_index::ANNIndex,
    index::{bruteforce_idx::BruteForceIndex, bruteforce_params::BruteForceParams},
};
use song_search::{SongEmbedding, EMBEDDING_DIMENSION};

#[tokio::main]
async fn main() -> Result<()> {
    let song_embeddings_path = "./data/output/song_embedding.yaml";

    // 1. Read all song embeddings
    let embeddings = std::fs::read_to_string(song_embeddings_path)?;
    let song_embeddings: Vec<SongEmbedding> = serde_yaml::from_str(&embeddings)?;
    println!("Total embeddings: {}", song_embeddings.len());

    // 2. Create index
    let mut index = BruteForceIndex::new(EMBEDDING_DIMENSION, &BruteForceParams::default());

    // 3. Add all song embeddings to index
    for (idx, song_embedding) in song_embeddings.iter().enumerate() {
        println!(
            "Adding Song {} by {} to index",
            song_embedding.title, song_embedding.artist
        );
        index
            .add(&song_embedding.embedding, idx)
            .map_err(|e| anyhow!(e))?;
    }

    // 4. Build index
    index
        .build(hora::core::metrics::Metric::Euclidean)
        .map_err(|e| anyhow!(e))?;
    println!("Index built.");

    let openai_client = Client::new();
    loop {
        // 5. Ask user for search query
        print!("Query: ");
        std::io::stdout().flush()?;
        let mut search = String::new();
        std::io::stdin().read_line(&mut search)?;

        // 6. Get embedding for search words
        let response = openai_client
            .embeddings()
            .create(
                CreateEmbeddingRequestArgs::default()
                    .input(search.trim().to_lowercase())
                    .model("text-embedding-ada-002")
                    .build()?,
            )
            .await?;

        // only one response expected
        let embedding = response.data.into_iter().nth(0).unwrap().embedding;

        // 7. Search Index for 5 nearest
        let searched = index.search(&embedding, 5);

        // 8. Display result to user
        for found in searched {
            println!("{}", song_embeddings[found]);
        }
    }
}
