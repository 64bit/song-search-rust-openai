use std::{fmt::Display, path::Path};

use ansi_term::Style;
use anyhow::Result;
use async_openai::{
    types::{CreateEmbeddingRequestArgs, CreateEmbeddingResponse},
    Client,
};
use serde::Deserialize;
use sqlx::{PgPool, Row};
use tracing::info;

pub const EMBEDDING_DIMENSION: usize = 1536;
pub const EMBEDDING_MODEL: &str = "text-embedding-ada-002";

#[derive(Debug, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub struct Song {
    pub artist: String,
    pub title: String,
    pub album: String,
    pub lyric: String,
}

impl Display for Song {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.album.is_empty() {
            write!(
                f,
                "{} by {}",
                Style::new().bold().paint(&self.title),
                Style::new().dimmed().paint(&self.artist)
            )
        } else {
            write!(
                f,
                "{} by {} / {}",
                Style::new().bold().paint(&self.title),
                Style::new().dimmed().paint(&self.artist),
                Style::new().italic().paint(&self.album)
            )
        }
    }
}

impl Song {
    /// Create extension, table and index
    pub async fn create_db_resources(pg_pool: &PgPool) -> Result<()> {
        // Create Extension
        let extension_query = "CREATE EXTENSION IF NOT EXISTS vector;";
        sqlx::query(extension_query).execute(pg_pool).await?;

        // Create Table
        let table_query = r#"CREATE TABLE IF NOT EXISTS songs
(
    artist text,
    title text,
    album text,
    lyric text,
    embedding vector(1536)
);"#;

        sqlx::query(table_query).execute(pg_pool).await?;

        // Create Index
        let index_query =
            "CREATE INDEX IF NOT EXISTS songs_idx ON songs USING ivfflat (embedding vector_cosine_ops);";

        sqlx::query(index_query).execute(pg_pool).await?;

        Ok(())
    }

    //// Read csv at given path into Vec<Song>
    pub fn get_songs<P: AsRef<Path>>(path: P) -> Result<Vec<Song>> {
        let mut rdr = csv::Reader::from_path(path)?;
        let mut songs = vec![];
        for result in rdr.deserialize() {
            let song: Song = result?;
            songs.push(song);
        }
        Ok(songs)
    }

    /// Append artist + title + album + lyrics as input text for
    /// Song embedding
    pub fn embedding_text(&self) -> String {
        [&self.artist, &self.title, &self.album, &self.lyric]
            .map(|s| s.to_owned())
            .join(" ")
            .replace("\n", " ")
            .trim()
            .to_lowercase()
    }

    /// Get embedding from OpenAI for this Song
    pub async fn get_embedding(&self, openai_client: &Client) -> Result<CreateEmbeddingResponse> {
        let response = openai_client
            .embeddings()
            .create(
                CreateEmbeddingRequestArgs::default()
                    .input(self.embedding_text())
                    .model(EMBEDDING_MODEL)
                    .build()?,
            )
            .await?;

        info!(
            "Song {} by {} used {} tokens",
            self.title, self.artist, response.usage.total_tokens
        );

        Ok(response)
    }

    /// Save embedding for this Song in DB
    pub async fn save_embedding(&self, pg_pool: &PgPool, pgvector: pgvector::Vector) -> Result<()> {
        sqlx::query("INSERT INTO songs (artist, title, album, lyric, embedding) VALUES ($1, $2, $3, $4, $5)")
            .bind(self.artist.clone())
            .bind(self.title.clone())
            .bind(self.album.clone())
            .bind(self.lyric.clone())
            .bind(pgvector)
            .execute(pg_pool)
            .await?;

        Ok(())
    }

    /// Search `n` nearest neighbors for given query in DB
    pub async fn query(query: &str, n: i8, client: &Client, pg_pool: &PgPool) -> Result<Vec<Song>> {
        let query = query.trim().to_lowercase();

        // Get embedding from OpenAI
        let response = client
            .embeddings()
            .create(
                CreateEmbeddingRequestArgs::default()
                    .input(query)
                    .model(EMBEDDING_MODEL)
                    .build()?,
            )
            .await?;

        let pgvector = pgvector::Vector::from(response.data[0].embedding.clone());

        // Search for nearest neighbors in database
        Ok(sqlx::query(
            "SELECT artist, title, album, lyric FROM songs ORDER BY embedding <-> $1 LIMIT $2::int",
        )
        .bind(pgvector)
        .bind(n)
        .fetch_all(pg_pool)
        .await?
        .into_iter()
        .map(|r| Song {
            artist: r.get("artist"),
            title: r.get("title"),
            album: r.get("album"),
            lyric: r.get("lyric"),
        })
        .collect())
    }
}
