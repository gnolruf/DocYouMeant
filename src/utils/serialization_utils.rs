use geo::Coord;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

#[derive(Serialize, Deserialize)]
struct CoordHelper {
    x: i32,
    y: i32,
}

pub mod coord_array_i32 {
    use super::*;

    pub fn serialize<S>(coords: &[Coord<i32>; 4], serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let helpers: Vec<CoordHelper> = coords
            .iter()
            .map(|c| CoordHelper { x: c.x, y: c.y })
            .collect();
        helpers.serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<[Coord<i32>; 4], D::Error>
    where
        D: Deserializer<'de>,
    {
        let helpers: Vec<CoordHelper> = Vec::deserialize(deserializer)?;
        if helpers.len() != 4 {
            return Err(serde::de::Error::custom("Expected 4 coordinates"));
        }
        Ok([
            Coord {
                x: helpers[0].x,
                y: helpers[0].y,
            },
            Coord {
                x: helpers[1].x,
                y: helpers[1].y,
            },
            Coord {
                x: helpers[2].x,
                y: helpers[2].y,
            },
            Coord {
                x: helpers[3].x,
                y: helpers[3].y,
            },
        ])
    }
}

pub mod coord_vec_i32 {
    use super::*;

    pub fn serialize<S>(coords: &[Coord<i32>], serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let helpers: Vec<CoordHelper> = coords
            .iter()
            .map(|c| CoordHelper { x: c.x, y: c.y })
            .collect();
        helpers.serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Vec<Coord<i32>>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let helpers: Vec<CoordHelper> = Vec::deserialize(deserializer)?;
        Ok(helpers
            .into_iter()
            .map(|h| Coord { x: h.x, y: h.y })
            .collect())
    }
}
