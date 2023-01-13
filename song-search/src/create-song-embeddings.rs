use anyhow::Result;
use async_openai::{
    types::{CreateEmbeddingRequestArgs, Embedding},
    Client,
};
use song_search::{Song, SongEmbedding};

#[tokio::main]
async fn main() -> Result<()> {
    let song_paths = [
        "./data/input/ColdPlay.csv",
        "./data/input/EdSheeran.csv",
        "./data/input/JustinBieber.csv",
        "./data/input/SelenaGomez.csv",
    ];
    let song_embeddings_path = "./data/output/song_embedding.yaml";

    // 1. Create OpenAI Client
    let openai_client = Client::new();

    // 2. Read all songs from csv file
    let mut songs: Vec<Song> = vec![];
    for path in song_paths {
        let mut s = Song::get_songs(path)?;
        println!("Read {} songs at {}", s.len(), path);
        songs.append(&mut s);
    }
    println!("Total song count: {}", songs.len());

    let mut output: Vec<SongEmbedding> = vec![];

    // 3. Make OpenAI call to get embedding for each song and add it to output
    for song in songs {
        let response = openai_client
            .embeddings()
            .create(
                CreateEmbeddingRequestArgs::default()
                    .input(song.embedding_text())
                    .model("text-embedding-ada-002")
                    .build()?,
            )
            .await?;

        // Tokens used by this Song
        println!(
            "Song {} by {} used {} tokens",
            &song.title, &song.artist, response.usage.total_tokens
        );

        // only one embedding expected
        let embedding: Embedding = response.data.into_iter().nth(0).unwrap();

        // 4. Create representation to save in the file
        let song_embedding = SongEmbedding {
            artist: song.artist,
            album: song.album,
            title: song.title,
            embedding: embedding.embedding,
        };

        output.push(song_embedding);
    }

    // 5. Save all embeddings to a file
    let yaml_file = std::fs::OpenOptions::new()
        .write(true)
        .create(true)
        .open(song_embeddings_path)?;
    serde_yaml::to_writer(yaml_file, &output)?;

    Ok(())
}
