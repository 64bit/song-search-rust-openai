use std::{fmt::Display, path::Path};

use anyhow::Result;
use serde::{Deserialize, Serialize};

pub const EMBEDDING_DIMENSION: usize = 1536;

#[derive(Debug, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub struct Song {
    pub artist: String,
    pub title: String,
    pub album: String,
    pub lyric: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SongEmbedding {
    pub artist: String,
    pub title: String,
    pub album: String,
    pub embedding: Vec<f32>,
}

impl Display for SongEmbedding {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "\t{} by {}", self.title, self.artist)
    }
}

impl Song {
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

    /// Append artist + title + album + lyrics and trim it to length of 8192
    pub fn embedding_text(&self) -> String {
        [&self.artist, &self.title, &self.album, &self.lyric]
            .map(|s| s.to_owned())
            .join(" ")
            .replace("\n", " ")
            .to_lowercase()
        //.chars()
        //.take(8192)
        //.collect()
    }
}
