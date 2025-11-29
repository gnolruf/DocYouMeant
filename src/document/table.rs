use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TableType {
    /// A table with visible grid lines/borders
    Wired,
    /// A table without visible grid lines/borders
    Wireless,
}
