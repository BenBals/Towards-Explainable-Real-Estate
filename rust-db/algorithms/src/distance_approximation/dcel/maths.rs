use std::{
    borrow::Borrow,
    fmt::{Debug, Display},
    ops::{Add, Index, IndexMut, Mul, Neg, Sub},
};

use num::{rational::Ratio, traits::NumRef, Integer, Zero};

/// This trait abstracts the different number types which can be used for Dcel.
pub trait DcelNum: NumRef + PartialOrd + Clone + Debug {
    fn dcel_eq(&self, other: &Self) -> bool {
        self == other
    }

    fn dcel_geq(&self, other: &Self) -> bool {
        self > other || self.dcel_eq(other)
    }

    fn dcel_greater(&self, other: &Self) -> bool {
        self > other && !self.dcel_eq(other)
    }

    fn dcel_leq(&self, other: &Self) -> bool {
        self < other || self.dcel_eq(other)
    }

    fn dcel_less(&self, other: &Self) -> bool {
        self < other && !self.dcel_eq(other)
    }

    fn dcel_non_negative(&self) -> bool {
        self.dcel_geq(&Self::zero())
    }

    fn dcel_non_positive(&self) -> bool {
        self.dcel_leq(&Self::zero())
    }

    /// Might change the number such that a vector with the given squared norm has norm one.
    /// It is not required to be done, if the result is not representable.
    fn normalize_by_squared_norm(self, squared_norm: &Self) -> Self;
}

impl DcelNum for f64 {
    fn dcel_eq(&self, other: &Self) -> bool {
        const EPS: f64 = 1e-9;

        let check_relative = self.abs() >= EPS && other.abs() >= EPS;

        if (self - other).abs() < EPS {
            true
        } else if check_relative {
            ((self - other) / self).abs() < EPS || ((self - other) / other).abs() < EPS
        } else {
            false
        }
    }

    fn normalize_by_squared_norm(self, squared_norm: &Self) -> Self {
        self / squared_norm.sqrt()
    }
}

impl<N: Clone + Integer + Debug> DcelNum for Ratio<N> {
    fn normalize_by_squared_norm(self, _squared_norm: &Self) -> Self {
        // don't do anything as the result might not be representable
        self
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Vector3<N: DcelNum>([N; 3]);

impl<N: DcelNum + Display> Display for Vector3<N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Vector3 ")?;
        self.0.fmt(f)
    }
}

impl<N: DcelNum> Vector3<N> {
    pub fn new(x: N, y: N, z: N) -> Self {
        Self([x, y, z])
    }

    pub fn dot(&self, other: &Self) -> N {
        self.0[0].clone() * &other.0[0]
            + self.0[1].clone() * &other.0[1]
            + self.0[2].clone() * &other.0[2]
    }

    pub fn squared_norm(&self) -> N {
        self.dot(self)
    }

    pub fn normalize(self) -> Self {
        let sq_norm = self.squared_norm();
        let [a, b, c] = self.0;
        Self::new(
            a.normalize_by_squared_norm(&sq_norm),
            b.normalize_by_squared_norm(&sq_norm),
            c.normalize_by_squared_norm(&sq_norm),
        )
    }

    pub fn cross(&self, other: &Self) -> Self {
        Self::new(
            self.0[1].clone() * &other.0[2] - self.0[2].clone() * &other.0[1],
            self.0[2].clone() * &other.0[0] - self.0[0].clone() * &other.0[2],
            self.0[0].clone() * &other.0[1] - self.0[1].clone() * &other.0[0],
        )
    }

    pub fn dcel_eq(&self, other: &Self) -> bool {
        self.0.iter().zip(other.0.iter()).all(|(a, b)| a.dcel_eq(b))
    }
}

impl<N: DcelNum> Default for Vector3<N> {
    fn default() -> Self {
        Self::new(N::zero(), N::zero(), N::zero())
    }
}

impl<N: DcelNum> Zero for Vector3<N> {
    fn zero() -> Self {
        Self::default()
    }

    fn is_zero(&self) -> bool {
        self.dcel_eq(&Self::zero())
    }
}

impl<N: DcelNum> Neg for Vector3<N> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        let [a, b, c] = self.0;
        Self::new(N::zero() - a, N::zero() - b, N::zero() - c)
    }
}

impl<N: DcelNum, V: Borrow<Vector3<N>>> Add<V> for Vector3<N> {
    type Output = Self;

    fn add(self, rhs: V) -> Self::Output {
        Self::new(
            self.0[0].clone() + &rhs.borrow().0[0],
            self.0[1].clone() + &rhs.borrow().0[1],
            self.0[2].clone() + &rhs.borrow().0[2],
        )
    }
}

impl<N: DcelNum, V: Borrow<Vector3<N>>> Sub<V> for Vector3<N> {
    type Output = Self;

    fn sub(self, rhs: V) -> Self::Output {
        Self::new(
            self.0[0].clone() - &rhs.borrow().0[0],
            self.0[1].clone() - &rhs.borrow().0[1],
            self.0[2].clone() - &rhs.borrow().0[2],
        )
    }
}

impl<N: DcelNum, S: Borrow<N>> Mul<S> for Vector3<N> {
    type Output = Self;

    fn mul(self, rhs: S) -> Self::Output {
        Self::new(
            self.0[0].clone() * rhs.borrow(),
            self.0[1].clone() * rhs.borrow(),
            self.0[2].clone() * rhs.borrow(),
        )
    }
}

impl<N: DcelNum> Index<usize> for Vector3<N> {
    type Output = N;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl<N: DcelNum> IndexMut<usize> for Vector3<N> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}
