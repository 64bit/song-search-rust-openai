use std::{path::PathBuf, sync::Arc, time::Duration};

use anyhow::Result;
use async_openai::Client;
use backoff::ExponentialBackoffBuilder;
use song_search::Song;
use sqlx::postgres::PgPoolOptions;
use tokio::sync::Semaphore;
use tracing::{error, info};
use tracing_subscriber::{fmt, prelude::*, EnvFilter};

#[tokio::main]
async fn main() -> Result<()> {
    std::env::set_var("RUST_LOG", "info,sqlx=off,async_openai=warn"); // ideally set outside the program

    // Setup Logging
    tracing_subscriber::registry()
        .with(fmt::layer())
        .with(EnvFilter::from_default_env())
        .init();

    // 1. Create OpenAI Client
    let backoff = ExponentialBackoffBuilder::new()
        .with_initial_interval(Duration::from_secs(15))
        .with_randomization_factor(0.1)
        .with_multiplier(3.0)
        .build();

    let openai_client = Client::new().with_backoff(backoff);

    // 2. Create PgPool for DB
    let pg_pool = PgPoolOptions::new()
        .max_connections(1500)
        .connect("postgres://async-openai:async-openai@localhost:5432/songs")
        .await?;

    // 3. Create DB resources
    Song::create_db_resources(&pg_pool).await?;

    // 4. Get all paths for all Artist data files
    let artist_files: Vec<PathBuf> = std::fs::read_dir("./data/input")?
        .map(|entry| entry.unwrap())
        .map(|entry| entry.path())
        .collect();

    // 5. Process each Artist file
    let mut total_songs = 0;
    let mut handles = vec![];
    // Only allow 5 tasks at a time
    let semaphore = Arc::new(Semaphore::new(5));
    for artist_file in artist_files {
        let songs = Song::get_songs(&artist_file)?;
        total_songs += songs.len();
        info!(
            "Processing songs {} in {}",
            songs.len(),
            artist_file.display()
        );
        for song in songs {
            let client = openai_client.clone();
            let pool = pg_pool.clone();
            // Only allow limit in-flight request at a time to "rate limit" API and DB calls and limit spawned tasks.
            let permit = semaphore.clone().acquire_owned().await?;
            let handle = tokio::spawn(async move {
                // 6. If song doesn't exist in DB, get its
                //    embedding and save it to DB
                if sqlx::query(
                    "SELECT * FROM songs WHERE artist = $1 AND title = $2 AND album = $3 LIMIT 1",
                )
                .bind(song.artist.clone())
                .bind(song.title.clone())
                .bind(song.album.clone())
                .fetch_one(&pool)
                .await
                .is_err()
                {
                    info!(
                        "Getting and saving embedding for {} by {}",
                        &song.title, &song.artist
                    );

                    let _ = song
                        .get_and_save_embedding(&client, &pool)
                        .await
                        .map_err(|e| {
                            error!(
                                "Failed to get and save embedding for {} by {}: {e}",
                                song.title, song.artist
                            )
                        });
                }
                drop(permit);
            });
            handles.push(handle);
        }
    }
    info!("Total Songs: {total_songs}");
    futures::future::join_all(handles).await;
    info!("Done");

    Ok(())
}
