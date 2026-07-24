/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Query-dependent graph start points ("IVF-lite").
//!
//! A DiskANN graph search normally begins from a small, fixed set of start points (e.g. the
//! medoid). This crate provides an optional coarse *router*: after the graph is built, run
//! k-means over the data to obtain `k` centroids, and remember, for each centroid, the graph
//! vertex closest to it (its *entry vertex*). At query time the caller looks up the `m`
//! nearest centroids and seeds the graph search from their entry vertices instead of the
//! fixed medoid.
//!
//! The centroids are kept in full precision: they only act as a coarse router (a handful of
//! comparisons per query), so there is no benefit to quantizing them, and doing so would
//! couple this crate to the PQ codebook. The fine-grained distance computations during graph
//! traversal are unaffected and remain whatever the index itself uses (e.g. PQ).
//!
//! # Vertex identifiers
//!
//! [`StartPointTable`] refers to graph vertices by the *row index* of the training data: row
//! `i` of the matrix passed to [`StartPointTable::build`] is assumed to correspond to graph
//! vertex `i`. Callers that use a different identifier scheme must translate accordingly.
//!
//! # Example
//!
//! ```
//! use std::num::NonZeroUsize;
//! use diskann_startpoints::{StartPointTable, StartPointsConfig};
//! use diskann_utils::views::MatrixView;
//! use diskann_vector::distance::Metric;
//!
//! let nz = |v| NonZeroUsize::new(v).unwrap();
//! // Two well-separated clusters in 2D (rows are vertices 0..=3).
//! let data = [0.0, 0.0, 0.1, 0.1, 10.0, 10.0, 10.1, 10.1];
//! let view = MatrixView::try_from(&data[..], 4, 2).unwrap();
//!
//! let config = StartPointsConfig::new(nz(2), nz(1), nz(16), 0, Metric::L2);
//! let table = StartPointTable::build(view, &config).unwrap().unwrap();
//!
//! // A query near the first cluster is routed to a vertex in that cluster.
//! let seeds = table.entry_points(&[0.05, 0.05]);
//! assert!(seeds.iter().all(|&v| v < 2));
//! ```

#![forbid(unsafe_code)]

use std::{mem::size_of, num::NonZeroUsize};

use diskann_quantization::algorithms::kmeans::{
    lloyds::lloyds,
    plusplus::{kmeans_plusplus_into, KMeansPlusPlusError},
};
use diskann_utils::views::{Matrix, MatrixView};
use diskann_vector::distance::{DistanceProvider, Metric};
use rand::{rngs::StdRng, SeedableRng};
use thiserror::Error;

const FILE_MAGIC: [u8; 8] = *b"DANNSTPT";
const FILE_VERSION: u32 = 1;
const FILE_HEADER_LEN: usize =
    FILE_MAGIC.len() + size_of::<u32>() + size_of::<i32>() + 3 * size_of::<u64>();

/// Configuration controlling how the start-point router is built and queried.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StartPointsConfig {
    /// Whether the router is active. When `false`, [`StartPointTable::build`] returns
    /// `Ok(None)` and the caller should fall back to the default (fixed) start points.
    pub enabled: bool,
    /// Number of k-means clusters / centroids (`k`). Chosen once, at build time. A common
    /// choice is `k ≈ sqrt(N)` for `N` vectors.
    pub num_clusters: NonZeroUsize,
    /// Number of nearest centroids to probe per query (`m`). Their entry vertices are used
    /// to seed the graph search. Typically a small value (1–8).
    pub num_probes: NonZeroUsize,
    /// Maximum number of Lloyd's iterations to run during clustering.
    pub max_iters: NonZeroUsize,
    /// Seed for the k-means initialization RNG, for reproducible builds.
    pub seed: u64,
    /// Distance metric. Must match the metric used by the index being routed.
    pub metric: Metric,
}

impl StartPointsConfig {
    /// Create an *enabled* configuration.
    pub fn new(
        num_clusters: NonZeroUsize,
        num_probes: NonZeroUsize,
        max_iters: NonZeroUsize,
        seed: u64,
        metric: Metric,
    ) -> Self {
        Self {
            enabled: true,
            num_clusters,
            num_probes,
            max_iters,
            seed,
            metric,
        }
    }

    /// Create a *disabled* configuration for `metric`. [`StartPointTable::build`] will
    /// return `Ok(None)`.
    pub fn disabled(metric: Metric) -> Self {
        Self {
            enabled: false,
            num_clusters: NonZeroUsize::MIN,
            num_probes: NonZeroUsize::MIN,
            max_iters: NonZeroUsize::MIN,
            seed: 0,
            metric,
        }
    }
}

/// Errors that can occur while building a [`StartPointTable`].
#[derive(Debug, Error)]
pub enum BuildError {
    /// The training data had zero rows or zero columns.
    #[error("training data is empty")]
    EmptyData,
    /// k-means initialization failed unrecoverably (e.g. the data contained NaN/infinity).
    #[error("k-means initialization failed: {0}")]
    Init(#[from] KMeansPlusPlusError),
    /// Clustering produced no non-empty clusters, so there are no entry vertices.
    #[error("k-means produced no non-empty clusters")]
    NoEntryVertices,
}

/// Errors that can occur while encoding or decoding a [`StartPointTable`].
#[derive(Debug, Error, PartialEq, Eq)]
pub enum PersistenceError {
    /// The serialized data does not begin with the start-point table magic bytes.
    #[error("invalid start-point table magic")]
    InvalidMagic,
    /// The serialized format version is not supported.
    #[error("unsupported start-point table version {0}")]
    UnsupportedVersion(u32),
    /// The serialized metric discriminant is invalid.
    #[error("invalid start-point table metric {0}")]
    InvalidMetric(i32),
    /// A required table dimension was zero.
    #[error("start-point table dimension must be non-zero")]
    ZeroDimension,
    /// The serialized table contained no centroids.
    #[error("start-point table must contain at least one centroid")]
    NoCentroids,
    /// The serialized probe count was zero.
    #[error("start-point table probe count must be non-zero")]
    ZeroProbes,
    /// Serialized dimensions could not be represented on this platform.
    #[error("start-point table dimensions exceed platform limits")]
    SizeOverflow,
    /// The payload length does not match the dimensions encoded in its header.
    #[error("invalid start-point table length: expected {expected} bytes, got {actual}")]
    InvalidLength {
        /// Length implied by the header.
        expected: usize,
        /// Actual payload length.
        actual: usize,
    },
    /// The in-memory centroid storage does not match the table dimensions.
    #[error("invalid centroid layout: expected {expected} coordinate values, got {actual}")]
    InvalidCentroidLayout {
        /// Coordinate count implied by the table dimensions.
        expected: usize,
        /// Actual coordinate count.
        actual: usize,
    },
    /// A serialized centroid coordinate was not finite.
    #[error("start-point table contains a non-finite centroid")]
    NonFiniteCentroid,
}

/// A `centroid -> entry vertex` table used to pick query-dependent graph start points.
#[derive(Debug, Clone, PartialEq)]
pub struct StartPointTable {
    /// Vector dimensionality.
    dim: usize,
    /// Row-major centroids (`num_centroids * dim`), aligned with `entry_vertices`.
    centroids: Vec<f32>,
    /// The graph vertex closest to each centroid.
    entry_vertices: Vec<u32>,
    /// Metric used for query-to-centroid comparisons.
    metric: Metric,
    /// Number of centroids to probe per query (`m`).
    num_probes: usize,
}

fn routing_metric(metric: Metric) -> Metric {
    match metric {
        // Lloyd's centroids are arithmetic means and are not generally unit vectors.
        Metric::CosineNormalized => Metric::Cosine,
        metric => metric,
    }
}

fn nearest_entry_vertices(
    data: MatrixView<'_, f32>,
    centroids: MatrixView<'_, f32>,
    assignments: &[u32],
    metric: Metric,
) -> Vec<Option<u32>> {
    let num_clusters = centroids.nrows();
    let mut non_empty = vec![false; num_clusters];
    for &cluster_id in assignments {
        non_empty[cluster_id as usize] = true;
    }

    let comparer = <f32 as DistanceProvider<f32>>::distance_comparer(metric, Some(data.ncols()));
    let mut best = vec![f32::INFINITY; num_clusters];
    let mut entry = vec![None; num_clusters];

    for (point_id, point) in data.row_iter().enumerate() {
        for (cluster, centroid) in centroids.row_iter().enumerate() {
            if !non_empty[cluster] {
                continue;
            }

            let distance = comparer.call(point, centroid);
            if distance < best[cluster] {
                best[cluster] = distance;
                entry[cluster] = Some(point_id as u32);
            }
        }
    }

    entry
}

impl StartPointTable {
    /// Build the router from `data` (row `i` corresponds to graph vertex `i`).
    ///
    /// Returns `Ok(None)` when `config.enabled` is `false`.
    ///
    /// `config.num_clusters` is clamped to the number of rows in `data`, and empty clusters
    /// are dropped, so the resulting table may contain fewer than `k` centroids.
    ///
    /// # Errors
    ///
    /// See [`BuildError`].
    pub fn build(
        data: MatrixView<'_, f32>,
        config: &StartPointsConfig,
    ) -> Result<Option<Self>, BuildError> {
        if !config.enabled {
            return Ok(None);
        }

        let num_points = data.nrows();
        let dim = data.ncols();
        if num_points == 0 || dim == 0 {
            return Err(BuildError::EmptyData);
        }

        // k-means++ requires at least as many points as clusters.
        let num_clusters = config.num_clusters.get().min(num_points);

        let mut centroids = Matrix::new(0.0f32, num_clusters, dim);
        let mut rng = StdRng::seed_from_u64(config.seed);

        match kmeans_plusplus_into(centroids.as_mut_view(), data, &mut rng) {
            Ok(()) => {}
            // Recoverable failures (e.g. insufficient diversity) still leave usable centers;
            // Lloyd's iterations and empty-cluster pruning below handle the remainder.
            Err(err) if err.is_numerically_recoverable() => {}
            Err(err) => return Err(BuildError::Init(err)),
        }

        let (assignments, _residual) =
            lloyds(data, centroids.as_mut_view(), config.max_iters.get());

        // Lloyd's returns assignments from before its final centroid update. Use them only
        // to identify non-empty clusters, then find each final centroid's nearest data point.
        let metric = routing_metric(config.metric);
        let entry =
            nearest_entry_vertices(data, centroids.as_view(), assignments.as_slice(), metric);

        // Keep only non-empty clusters, keeping centroids and entry vertices aligned.
        let mut kept_centroids: Vec<f32> = Vec::with_capacity(num_clusters * dim);
        let mut entry_vertices: Vec<u32> = Vec::with_capacity(num_clusters);
        for (centroid, maybe_vertex) in centroids.as_view().row_iter().zip(&entry) {
            if let Some(vertex) = *maybe_vertex {
                kept_centroids.extend_from_slice(centroid);
                entry_vertices.push(vertex);
            }
        }

        if entry_vertices.is_empty() {
            return Err(BuildError::NoEntryVertices);
        }

        Ok(Some(Self {
            dim,
            centroids: kept_centroids,
            entry_vertices,
            metric,
            num_probes: config.num_probes.get(),
        }))
    }

    /// Return the entry vertices of the `m` centroids nearest to `query`, closest first.
    ///
    /// The returned slice has at most `min(m, num_centroids)` elements and is suitable for
    /// use as graph-search seed points.
    ///
    /// # Panics
    ///
    /// Panics if `query.len()` does not equal [`Self::dim`].
    pub fn entry_points(&self, query: &[f32]) -> Vec<u32> {
        assert_eq!(
            query.len(),
            self.dim,
            "query dimension ({}) must match centroid dimension ({})",
            query.len(),
            self.dim
        );

        let comparer =
            <f32 as DistanceProvider<f32>>::distance_comparer(self.metric, Some(self.dim));
        let mut scored: Vec<(f32, u32)> = self
            .centroids
            .chunks_exact(self.dim)
            .zip(self.entry_vertices.iter().copied())
            .map(|(centroid, vertex)| (comparer.call(query, centroid), vertex))
            .collect();

        let probes = self.num_probes.min(scored.len());
        if probes < scored.len() {
            scored.select_nth_unstable_by(probes - 1, |a, b| a.0.total_cmp(&b.0));
            scored.truncate(probes);
        }
        scored.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
        scored.into_iter().map(|(_, vertex)| vertex).collect()
    }

    /// Number of centroids retained in the table.
    pub fn num_centroids(&self) -> usize {
        self.entry_vertices.len()
    }

    /// Number of centroids probed per query (`m`).
    pub fn num_probes(&self) -> usize {
        self.num_probes
    }

    /// Vector dimensionality expected by [`Self::entry_points`].
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// The entry vertex for every retained centroid.
    pub fn entry_vertices(&self) -> &[u32] {
        &self.entry_vertices
    }

    /// Metric used to compare queries and centroids.
    pub fn metric(&self) -> Metric {
        self.metric
    }

    /// Serialize this table using the versioned DiskANN start-point format.
    ///
    /// # Errors
    ///
    /// Returns [`PersistenceError::SizeOverflow`] if the table is too large to encode.
    pub fn to_bytes(&self) -> Result<Vec<u8>, PersistenceError> {
        let num_centroids = self.entry_vertices.len();
        let centroid_values = num_centroids
            .checked_mul(self.dim)
            .ok_or(PersistenceError::SizeOverflow)?;
        let expected_values = self.centroids.len();
        if centroid_values != expected_values {
            return Err(PersistenceError::InvalidCentroidLayout {
                expected: centroid_values,
                actual: expected_values,
            });
        }

        let payload_len = centroid_values
            .checked_mul(size_of::<f32>())
            .and_then(|len| {
                num_centroids
                    .checked_mul(size_of::<u32>())
                    .and_then(|entries_len| len.checked_add(entries_len))
            })
            .ok_or(PersistenceError::SizeOverflow)?;
        let capacity = FILE_HEADER_LEN
            .checked_add(payload_len)
            .ok_or(PersistenceError::SizeOverflow)?;

        let mut bytes = Vec::with_capacity(capacity);
        bytes.extend_from_slice(&FILE_MAGIC);
        bytes.extend_from_slice(&FILE_VERSION.to_le_bytes());
        bytes.extend_from_slice(&i32::from(self.metric).to_le_bytes());
        bytes.extend_from_slice(
            &u64::try_from(self.dim)
                .map_err(|_| PersistenceError::SizeOverflow)?
                .to_le_bytes(),
        );
        bytes.extend_from_slice(
            &u64::try_from(num_centroids)
                .map_err(|_| PersistenceError::SizeOverflow)?
                .to_le_bytes(),
        );
        bytes.extend_from_slice(
            &u64::try_from(self.num_probes)
                .map_err(|_| PersistenceError::SizeOverflow)?
                .to_le_bytes(),
        );
        for &value in &self.centroids {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        for &vertex in &self.entry_vertices {
            bytes.extend_from_slice(&vertex.to_le_bytes());
        }

        Ok(bytes)
    }

    /// Deserialize a table encoded by [`Self::to_bytes`].
    ///
    /// # Errors
    ///
    /// Returns [`PersistenceError`] when the header, dimensions, or payload are invalid.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, PersistenceError> {
        if bytes.len() < FILE_HEADER_LEN {
            return Err(PersistenceError::InvalidLength {
                expected: FILE_HEADER_LEN,
                actual: bytes.len(),
            });
        }
        if bytes[..FILE_MAGIC.len()] != FILE_MAGIC {
            return Err(PersistenceError::InvalidMagic);
        }

        let mut offset = FILE_MAGIC.len();
        let version = read_u32(bytes, &mut offset);
        if version != FILE_VERSION {
            return Err(PersistenceError::UnsupportedVersion(version));
        }

        let metric_raw = read_i32(bytes, &mut offset);
        let metric = Metric::try_from(metric_raw)
            .map_err(|_| PersistenceError::InvalidMetric(metric_raw))?;
        let dim = usize::try_from(read_u64(bytes, &mut offset))
            .map_err(|_| PersistenceError::SizeOverflow)?;
        let num_centroids = usize::try_from(read_u64(bytes, &mut offset))
            .map_err(|_| PersistenceError::SizeOverflow)?;
        let num_probes = usize::try_from(read_u64(bytes, &mut offset))
            .map_err(|_| PersistenceError::SizeOverflow)?;

        if dim == 0 {
            return Err(PersistenceError::ZeroDimension);
        }
        if num_centroids == 0 {
            return Err(PersistenceError::NoCentroids);
        }
        if num_probes == 0 {
            return Err(PersistenceError::ZeroProbes);
        }

        let centroid_values = num_centroids
            .checked_mul(dim)
            .ok_or(PersistenceError::SizeOverflow)?;
        let payload_len = centroid_values
            .checked_mul(size_of::<f32>())
            .and_then(|len| {
                num_centroids
                    .checked_mul(size_of::<u32>())
                    .and_then(|entries_len| len.checked_add(entries_len))
            })
            .ok_or(PersistenceError::SizeOverflow)?;
        let expected_len = FILE_HEADER_LEN
            .checked_add(payload_len)
            .ok_or(PersistenceError::SizeOverflow)?;
        if bytes.len() != expected_len {
            return Err(PersistenceError::InvalidLength {
                expected: expected_len,
                actual: bytes.len(),
            });
        }

        let mut centroids = Vec::with_capacity(centroid_values);
        for _ in 0..centroid_values {
            let value = f32::from_le_bytes(read_array(bytes, &mut offset));
            if !value.is_finite() {
                return Err(PersistenceError::NonFiniteCentroid);
            }
            centroids.push(value);
        }

        let mut entry_vertices = Vec::with_capacity(num_centroids);
        for _ in 0..num_centroids {
            entry_vertices.push(read_u32(bytes, &mut offset));
        }

        Ok(Self {
            dim,
            centroids,
            entry_vertices,
            metric,
            num_probes,
        })
    }
}

fn read_array<const N: usize>(bytes: &[u8], offset: &mut usize) -> [u8; N] {
    let end = *offset + N;
    let value = bytes[*offset..end]
        .try_into()
        .expect("payload length was validated");
    *offset = end;
    value
}

fn read_u32(bytes: &[u8], offset: &mut usize) -> u32 {
    u32::from_le_bytes(read_array(bytes, offset))
}

fn read_i32(bytes: &[u8], offset: &mut usize) -> i32 {
    i32::from_le_bytes(read_array(bytes, offset))
}

fn read_u64(bytes: &[u8], offset: &mut usize) -> u64 {
    u64::from_le_bytes(read_array(bytes, offset))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn nz(value: usize) -> NonZeroUsize {
        NonZeroUsize::new(value).expect("non-zero")
    }

    /// Two well-separated 2D clusters: vertices {0,1,2} near the origin, {3,4,5} near (10,10).
    const DATA: [f32; 12] = [
        0.0, 0.0, // 0
        0.1, 0.0, // 1
        0.0, 0.1, // 2
        10.0, 10.0, // 3
        10.1, 10.0, // 4
        10.0, 10.1, // 5
    ];

    fn view() -> MatrixView<'static, f32> {
        MatrixView::try_from(&DATA[..], 6, 2).expect("valid matrix")
    }

    #[test]
    fn disabled_config_returns_none() {
        let config = StartPointsConfig::disabled(Metric::L2);
        let table = StartPointTable::build(view(), &config).expect("build");
        assert!(table.is_none());
    }

    #[test]
    fn routes_query_to_nearest_cluster() {
        let config = StartPointsConfig::new(nz(2), nz(1), nz(32), 42, Metric::L2);
        let table = StartPointTable::build(view(), &config)
            .expect("build")
            .expect("enabled");

        assert_eq!(table.num_centroids(), 2);

        // Near the origin cluster -> a vertex in {0,1,2}.
        let near = table.entry_points(&[0.05, 0.05]);
        assert_eq!(near.len(), 1);
        assert!(
            near[0] < 3,
            "expected an origin-cluster vertex, got {}",
            near[0]
        );

        // Near the far cluster -> a vertex in {3,4,5}.
        let far = table.entry_points(&[9.9, 9.9]);
        assert_eq!(far.len(), 1);
        assert!(far[0] >= 3, "expected a far-cluster vertex, got {}", far[0]);
    }

    #[test]
    fn num_clusters_is_clamped_to_point_count() {
        // Ask for more clusters than there are points.
        let config = StartPointsConfig::new(nz(100), nz(4), nz(16), 7, Metric::L2);
        let table = StartPointTable::build(view(), &config)
            .expect("build")
            .expect("enabled");
        assert!(table.num_centroids() <= 6);
    }

    #[test]
    fn probes_are_clamped_to_centroid_count() {
        // m larger than the number of centroids: return one seed per centroid.
        let config = StartPointsConfig::new(nz(2), nz(8), nz(16), 1, Metric::L2);
        let table = StartPointTable::build(view(), &config)
            .expect("build")
            .expect("enabled");
        let seeds = table.entry_points(&[5.0, 5.0]);
        assert_eq!(seeds.len(), table.num_centroids());
    }

    #[test]
    fn entry_vertices_are_nearest_to_final_centroids() {
        let data = [0.0, 0.0, 10.0, 0.0];
        let data = MatrixView::try_from(&data[..], 2, 2).expect("valid data");
        let centroids = [9.0, 0.0, 10.0, 0.0];
        let centroids = MatrixView::try_from(&centroids[..], 2, 2).expect("valid centroids");

        // Point 1 was assigned to cluster 1 before the final update, but is now also the
        // closest point to cluster 0's final centroid.
        let entries = nearest_entry_vertices(data, centroids, &[0, 1], Metric::L2);
        assert_eq!(entries, vec![Some(1), Some(1)]);
    }

    #[test]
    fn normalized_cosine_uses_scale_independent_centroid_routing() {
        let table = StartPointTable {
            dim: 2,
            centroids: vec![10.0, 10.0, 1.0, 0.0],
            entry_vertices: vec![0, 1],
            metric: routing_metric(Metric::CosineNormalized),
            num_probes: 1,
        };

        assert_eq!(table.entry_points(&[1.0, 0.0]), vec![1]);
    }

    #[test]
    fn persistence_round_trip() {
        let table = StartPointTable {
            dim: 2,
            centroids: vec![1.0, 2.0, 3.0, 4.0],
            entry_vertices: vec![7, 11],
            metric: Metric::L2,
            num_probes: 2,
        };

        let encoded = table.to_bytes().expect("encode");
        assert_eq!(
            StartPointTable::from_bytes(&encoded).expect("decode"),
            table
        );
    }

    #[test]
    fn persistence_rejects_truncated_payload() {
        let table = StartPointTable {
            dim: 2,
            centroids: vec![1.0, 2.0],
            entry_vertices: vec![7],
            metric: Metric::L2,
            num_probes: 1,
        };
        let mut encoded = table.to_bytes().expect("encode");
        encoded.pop();

        assert!(matches!(
            StartPointTable::from_bytes(&encoded),
            Err(PersistenceError::InvalidLength { .. })
        ));
    }

    #[test]
    #[should_panic(expected = "query dimension")]
    fn entry_points_rejects_wrong_dimension() {
        let config = StartPointsConfig::new(nz(2), nz(1), nz(16), 0, Metric::L2);
        let table = StartPointTable::build(view(), &config)
            .expect("build")
            .expect("enabled");
        let _ = table.entry_points(&[1.0, 2.0, 3.0]);
    }
}
