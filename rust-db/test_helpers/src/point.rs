//! This module contains a Point struct which can be used for testing code consuming Pointlikes.

use common::{Keyed, Pointlike};

/// a basic type that implements Pointlike
/// mainly used for testing purposes
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Point {
    x: u64,
    y: u64,
    key: usize,
}

impl Pointlike for Point {
    fn x(&self) -> u64 {
        self.x
    }
    fn y(&self) -> u64 {
        self.y
    }
}

impl Keyed for Point {
    type Key = usize;
    fn key(&self) -> usize {
        self.key
    }
}

impl Point {
    /// create a new Point
    pub fn new(x: u64, y: u64, key: usize) -> Self {
        Self { x, y, key }
    }
}
