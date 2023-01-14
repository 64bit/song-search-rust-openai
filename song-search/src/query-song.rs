use std::io::Write;

use ansi_term::Colour::Yellow;
use anyhow::Result;
use async_openai::Client;
use song_search::Song;
use sqlx::postgres::PgPoolOptions;
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
    let openai_client = Client::new();

    // 2. Create PgPool for DB
    let pg_pool = PgPoolOptions::new()
        .connect("postgres://async-openai:async-openai@localhost:5432/songs")
        .await?;

    loop {
        // 3. Ask user for search query
        print!("\n{}: ", Yellow.underline().paint("Query"));
        std::io::stdout().flush()?;
        let mut search = String::new();
        std::io::stdin().read_line(&mut search)?;

        // 4. Get Embedding from OpenAI and Search for nearest neighbors in DB
        let songs = Song::query(&search, 10, &openai_client, &pg_pool).await?;

        // 5. Display result to user
        for song in songs {
            println!("{}", song);
        }
    }
}
