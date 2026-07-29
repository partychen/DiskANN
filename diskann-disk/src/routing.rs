/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Query-aware entry points for disk graph search.

use std::{
    collections::HashSet,
    io::{Read, Write},
    num::NonZeroUsize,
};

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use diskann::{error::IntoANNResult, utils::VectorRepr, ANNError, ANNResult};
use diskann_providers::{
    storage::{PQStorage, StorageReadProvider, StorageWriteProvider},
    utils::{gen_random_slice, load_metadata_from_file, RayonThreadPoolRef, VectorDataIterator},
};
use diskann_vector::{distance::Metric, DistanceFunction};
use rand::Rng;

use crate::utils::{compute_closest_centers, k_means_clustering, spherical_k_means_clustering};

const ROUTING_TABLE_MAGIC: &[u8; 8] = b"DANNRTE1";
const REPRESENTATIVE_BATCH_SIZE: usize = 1_200;

/// In-memory vectors and graph IDs used to choose query-specific graph entry points.
#[derive(Debug, Clone)]
pub struct RoutingTable {
    ids: Vec<u32>,
    vectors: Vec<f32>,
    dimension: usize,
}

impl RoutingTable {
    /// Construct a routing table from real graph node IDs and vectors.
    pub fn new(ids: Vec<u32>, vectors: Vec<f32>, dimension: usize) -> ANNResult<Self> {
        if ids.is_empty() {
            return Err(ANNError::log_index_config_error(
                "routing_entries".into(),
                "routing table must contain at least one entry".into(),
            ));
        }
        if dimension == 0 || vectors.len() != ids.len() * dimension {
            return Err(ANNError::log_index_config_error(
                "routing_dimension".into(),
                format!(
                    "routing vectors contain {} elements for {} entries with dimension {}",
                    vectors.len(),
                    ids.len(),
                    dimension
                ),
            ));
        }

        Ok(Self {
            ids,
            vectors,
            dimension,
        })
    }

    /// Number of routing entries.
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    /// Whether the routing table is empty.
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    /// Stored vector dimension.
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Real graph node IDs represented by this table.
    pub fn ids(&self) -> &[u32] {
        &self.ids
    }

    /// Select up to `count` unique graph IDs nearest to `query`.
    pub fn select(
        &self,
        query: &[f32],
        metric: Metric,
        count: NonZeroUsize,
    ) -> ANNResult<Vec<u32>> {
        if query.len() != self.dimension {
            return Err(ANNError::log_index_error(format_args!(
                "query dimension {} does not match routing dimension {}",
                query.len(),
                self.dimension
            )));
        }

        let distance = f32::distance(metric, Some(self.dimension));
        let mut ranked = self
            .ids
            .iter()
            .copied()
            .zip(self.vectors.chunks_exact(self.dimension))
            .map(|(id, vector)| (distance.evaluate_similarity(query, vector), id))
            .collect::<Vec<_>>();
        ranked.sort_unstable_by(|left, right| left.0.total_cmp(&right.0));

        let mut selected = Vec::with_capacity(count.get().min(ranked.len()));
        let mut seen = HashSet::with_capacity(selected.capacity());
        for (_, id) in ranked {
            if seen.insert(id) {
                selected.push(id);
                if selected.len() == count.get() {
                    break;
                }
            }
        }
        Ok(selected)
    }

    /// Load a routing table sidecar.
    pub fn load<P>(path: &str, storage_provider: &P) -> ANNResult<Self>
    where
        P: StorageReadProvider,
    {
        let mut reader = storage_provider.open_reader(path)?;
        let mut magic = [0u8; ROUTING_TABLE_MAGIC.len()];
        reader.read_exact(&mut magic)?;
        if &magic != ROUTING_TABLE_MAGIC {
            return Err(ANNError::log_invalid_file_format(format!(
                "'{}' is not a supported routing table",
                path
            )));
        }

        let num_entries = reader.read_u32::<LittleEndian>()? as usize;
        let dimension = reader.read_u32::<LittleEndian>()? as usize;
        let element_size = reader.read_u32::<LittleEndian>()? as usize;
        if element_size != std::mem::size_of::<f32>() {
            return Err(ANNError::log_invalid_file_format(format!(
                "routing table element size {} does not match requested type size {}",
                element_size,
                std::mem::size_of::<f32>()
            )));
        }

        let vector_size = dimension
            .checked_mul(element_size)
            .ok_or_else(|| ANNError::log_invalid_file_format("routing table size overflow"))?;
        let entry_size = std::mem::size_of::<u32>()
            .checked_add(vector_size)
            .ok_or_else(|| ANNError::log_invalid_file_format("routing table size overflow"))?;
        let entries_size = num_entries
            .checked_mul(entry_size)
            .ok_or_else(|| ANNError::log_invalid_file_format("routing table size overflow"))?;
        let expected_size = ROUTING_TABLE_MAGIC
            .len()
            .checked_add(3 * std::mem::size_of::<u32>())
            .and_then(|header_size| header_size.checked_add(entries_size))
            .ok_or_else(|| ANNError::log_invalid_file_format("routing table size overflow"))?;
        if storage_provider.get_length(path)? != expected_size as u64 {
            return Err(ANNError::log_invalid_file_format(format!(
                "routing table '{}' has an invalid length",
                path
            )));
        }

        let mut ids = Vec::with_capacity(num_entries);
        let mut vectors = Vec::with_capacity(num_entries * dimension);
        let mut vector = vec![0.0f32; dimension];
        for _ in 0..num_entries {
            ids.push(reader.read_u32::<LittleEndian>()?);
            reader.read_exact(bytemuck::cast_slice_mut(&mut vector))?;
            vectors.extend_from_slice(&vector);
        }

        Self::new(ids, vectors, dimension)
    }

    /// Save this routing table as a sidecar.
    pub fn save<P>(&self, path: &str, storage_provider: &P) -> ANNResult<()>
    where
        P: StorageWriteProvider,
    {
        let mut writer = storage_provider.create_for_write(path)?;
        writer.write_all(ROUTING_TABLE_MAGIC)?;
        writer.write_u32::<LittleEndian>(self.len().try_into().map_err(|_| {
            ANNError::log_index_error("routing table contains more than u32::MAX entries")
        })?)?;
        writer.write_u32::<LittleEndian>(self.dimension.try_into().map_err(|_| {
            ANNError::log_index_error("routing table dimension exceeds u32::MAX")
        })?)?;
        writer.write_u32::<LittleEndian>(std::mem::size_of::<f32>() as u32)?;
        for (id, vector) in self
            .ids
            .iter()
            .zip(self.vectors.chunks_exact(self.dimension))
        {
            writer.write_u32::<LittleEndian>(*id)?;
            writer.write_all(bytemuck::cast_slice(vector))?;
        }
        writer.flush()?;
        Ok(())
    }
}

/// Train L2 k-means centers and map each center to its nearest real dataset node.
#[allow(clippy::too_many_arguments)]
pub fn generate_routing_table<T, P>(
    dataset_path: &str,
    output_path: &str,
    num_centers: NonZeroUsize,
    sampling_rate: f64,
    max_kmeans_reps: NonZeroUsize,
    storage_provider: &P,
    rng: &mut impl Rng,
    pool: RayonThreadPoolRef<'_>,
) -> ANNResult<RoutingTable>
where
    T: VectorRepr,
    P: StorageReadProvider + StorageWriteProvider,
{
    if !(0.0 < sampling_rate && sampling_rate <= 1.0) {
        return Err(ANNError::log_index_config_error(
            "routing_sampling_rate".into(),
            format!("sampling rate must be in (0, 1], got {}", sampling_rate),
        ));
    }

    let (training_data, num_training_points, full_dimension) =
        gen_random_slice::<T, P>(dataset_path, sampling_rate, storage_provider, rng)?;
    if num_training_points < num_centers.get() {
        return Err(ANNError::log_index_config_error(
            "routing_num_centers".into(),
            format!(
                "{} sampled training points are insufficient for {} centers",
                num_training_points,
                num_centers.get()
            ),
        ));
    }

    let mut centers = vec![0.0; num_centers.get() * full_dimension];
    k_means_clustering(
        &training_data,
        num_training_points,
        full_dimension,
        &mut centers,
        num_centers.get(),
        max_kmeans_reps.get(),
        rng,
        &mut false,
        pool,
    )?;

    let mut dataset = VectorDataIterator::<P, T>::new(dataset_path, None, storage_provider)?;
    let num_points = dataset.get_num_points();
    let mut best_distances = vec![f32::INFINITY; num_centers.get()];
    let mut representative_ids = vec![0; num_centers.get()];
    let mut representative_vectors = vec![0.0; num_centers.get() * full_dimension];
    let mut processed = 0usize;

    while let Some(batch) = dataset.next_n(REPRESENTATIVE_BATCH_SIZE) {
        let batch_size = batch.len();
        let mut batch_f32 = vec![0.0; batch_size * full_dimension];
        for (row, (vector, ())) in batch.iter().enumerate() {
            T::as_f32_into(
                vector,
                &mut batch_f32[row * full_dimension..(row + 1) * full_dimension],
            )
            .into_ann_result()?;
        }

        let mut closest_centers = vec![0; batch_size];
        compute_closest_centers(
            &batch_f32,
            batch_size,
            full_dimension,
            &centers,
            num_centers.get(),
            1,
            &mut closest_centers,
            None,
            None,
            pool,
        )?;

        for (row, (_, center_id)) in batch.iter().zip(closest_centers).enumerate() {
            let center_id = center_id as usize;
            let point = &batch_f32[row * full_dimension..(row + 1) * full_dimension];
            let center = &centers[center_id * full_dimension..(center_id + 1) * full_dimension];
            let distance = point
                .iter()
                .zip(center)
                .map(|(left, right)| (left - right) * (left - right))
                .sum();
            if distance < best_distances[center_id] {
                best_distances[center_id] = distance;
                representative_ids[center_id] = (processed + row).try_into().map_err(|_| {
                    ANNError::log_index_error("dataset contains more than u32::MAX points")
                })?;
                representative_vectors
                    [center_id * full_dimension..(center_id + 1) * full_dimension]
                    .copy_from_slice(point);
            }
        }
        processed += batch_size;
    }

    if processed != num_points {
        return Err(ANNError::log_invalid_file_format(format!(
            "read {} of {} vectors while generating routing entries",
            processed, num_points
        )));
    }

    let table = RoutingTable::new(representative_ids, representative_vectors, full_dimension)?;
    table.save(output_path, storage_provider)?;
    Ok(table)
}

/// L2-normalize a vector in place. No-op for zero vectors.
fn normalize_in_place(v: &mut [f32]) {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        let inv = 1.0 / norm;
        for x in v.iter_mut() {
            *x *= inv;
        }
    }
}

/// Generate a routing table by clustering in the index's Search-PQ space.
///
/// Unlike [`generate_routing_table`] (L2 k-means over full-precision base vectors), this:
/// - reconstructs each base vector from the index's Search-PQ codes — the same approximation
///   search actually operates on;
/// - uses ordinary k-means for L2 or spherical k-means for inner product;
/// - selects each region's representative with the same metric used for clustering.
///
/// Stored representative vectors are the reconstructed (PQ-space) vectors, so query-time
/// [`RoutingTable::select`] scores entries in the same space search estimates distances in.
#[allow(clippy::too_many_arguments)]
pub fn generate_routing_table_pq<P>(
    pq_pivots_path: &str,
    pq_compressed_path: &str,
    output_path: &str,
    metric: Metric,
    num_centers: NonZeroUsize,
    sampling_rate: f64,
    max_kmeans_reps: NonZeroUsize,
    storage_provider: &P,
    rng: &mut impl Rng,
    pool: RayonThreadPoolRef<'_>,
) -> ANNResult<RoutingTable>
where
    P: StorageReadProvider + StorageWriteProvider,
{
    if !(0.0 < sampling_rate && sampling_rate <= 1.0) {
        return Err(ANNError::log_index_config_error(
            "routing_sampling_rate".into(),
            format!("sampling rate must be in (0, 1], got {}", sampling_rate),
        ));
    }

    // Load the index quantizer (pivots + compressed codes) to reconstruct the approximate
    // vectors that search operates on.
    let pq_storage = PQStorage::new(pq_pivots_path, pq_compressed_path, None);
    let pq_table = pq_storage.load_pq_pivots_bin(pq_pivots_path, 0, storage_provider)?;
    let metadata = load_metadata_from_file(storage_provider, pq_compressed_path)?;
    let num_points = metadata.npoints();
    let num_chunks = pq_table.get_num_chunks();
    let dim = pq_table.get_dim();
    let compressed = PQStorage::load_pq_compressed_vectors_bin(
        pq_compressed_path,
        num_points,
        num_chunks,
        storage_provider,
    )?;

    if num_points < num_centers.get() {
        return Err(ANNError::log_index_config_error(
            "routing_num_centers".into(),
            format!(
                "{num_points} points are insufficient for {} centers",
                num_centers.get()
            ),
        ));
    }

    let reconstruct = |id: usize, out: &mut [f32]| -> ANNResult<()> {
        let code = compressed.get_row(id).ok_or_else(|| {
            ANNError::log_index_error("vector id out of range in compressed data")
        })?;
        pq_table.inflate_vector_into(code, out);
        Ok(())
    };

    if !matches!(metric, Metric::L2 | Metric::InnerProduct) {
        return Err(ANNError::log_index_config_error(
            "routing_metric".into(),
            format!(
                "PQ-space routing supports only l2 and innerproduct, got {}",
                metric
            ),
        ));
    }

    // Training set: reconstruct a random sample. MIPS uses unit-normalized vectors for
    // spherical k-means; L2 preserves the reconstructed magnitudes.
    let num_train = (((num_points as f64) * sampling_rate).round() as usize)
        .max(num_centers.get())
        .min(num_points);
    let mut sampled: Vec<usize> = (0..num_points).collect();
    for i in 0..num_train {
        let j = rng.random_range(i..num_points);
        sampled.swap(i, j);
    }
    sampled.truncate(num_train);

    let mut training = vec![0.0f32; num_train * dim];
    let mut buf = vec![0.0f32; dim];
    for (row, &id) in sampled.iter().enumerate() {
        reconstruct(id, &mut buf)?;
        if metric == Metric::InnerProduct {
            normalize_in_place(&mut buf);
        }
        training[row * dim..(row + 1) * dim].copy_from_slice(&buf);
    }

    let mut centers = vec![0.0f32; num_centers.get() * dim];
    match metric {
        Metric::L2 => {
            k_means_clustering(
                &training,
                num_train,
                dim,
                &mut centers,
                num_centers.get(),
                max_kmeans_reps.get(),
                rng,
                &mut false,
                pool,
            )?;
        }
        Metric::InnerProduct => {
            spherical_k_means_clustering(
                &training,
                num_train,
                dim,
                &mut centers,
                num_centers.get(),
                max_kmeans_reps.get(),
                rng,
                &mut false,
                pool,
            )?;
        }
        _ => unreachable!("metric validated above"),
    }

    // Assign every base point and keep the closest real node to each center.
    let initial_score = match metric {
        Metric::L2 => f32::INFINITY,
        Metric::InnerProduct => f32::NEG_INFINITY,
        _ => unreachable!("metric validated above"),
    };
    let mut best_score = vec![initial_score; num_centers.get()];
    let mut rep_ids = vec![0u32; num_centers.get()];
    let mut rep_vectors = vec![0.0f32; num_centers.get() * dim];
    for id in 0..num_points {
        reconstruct(id, &mut buf)?;
        let mut best_c = 0usize;
        let mut best_value = initial_score;
        for c in 0..num_centers.get() {
            let center = &centers[c * dim..(c + 1) * dim];
            let value = match metric {
                Metric::L2 => buf.iter().zip(center).map(|(x, y)| (x - y) * (x - y)).sum(),
                Metric::InnerProduct => buf.iter().zip(center).map(|(x, y)| x * y).sum(),
                _ => unreachable!("metric validated above"),
            };
            let is_better = match metric {
                Metric::L2 => value < best_value,
                Metric::InnerProduct => value > best_value,
                _ => unreachable!("metric validated above"),
            };
            if is_better {
                best_value = value;
                best_c = c;
            }
        }
        let is_better_representative = match metric {
            Metric::L2 => best_value < best_score[best_c],
            Metric::InnerProduct => best_value > best_score[best_c],
            _ => unreachable!("metric validated above"),
        };
        if is_better_representative {
            best_score[best_c] = best_value;
            rep_ids[best_c] = id.try_into().map_err(|_| {
                ANNError::log_index_error("dataset contains more than u32::MAX points")
            })?;
            rep_vectors[best_c * dim..(best_c + 1) * dim].copy_from_slice(&buf);
        }
    }

    // Drop empty regions (rare for small k), keeping only populated representatives.
    let mut ids = Vec::with_capacity(num_centers.get());
    let mut vectors = Vec::with_capacity(num_centers.get() * dim);
    for c in 0..num_centers.get() {
        if best_score[c].is_finite() {
            ids.push(rep_ids[c]);
            vectors.extend_from_slice(&rep_vectors[c * dim..(c + 1) * dim]);
        }
    }

    let table = RoutingTable::new(ids, vectors, dim)?;
    table.save(output_path, storage_provider)?;
    Ok(table)
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;

    use diskann_providers::storage::VirtualStorageProvider;
    use diskann_vector::distance::Metric;

    use super::*;

    #[test]
    fn routing_table_round_trip_and_selection() {
        let storage = VirtualStorageProvider::new_memory();
        let table =
            RoutingTable::new(vec![10, 20, 30], vec![0.0f32, 0.0, 5.0, 5.0, 9.0, 9.0], 2).unwrap();
        table.save("/routing.bin", &storage).unwrap();

        let loaded = RoutingTable::load("/routing.bin", &storage).unwrap();
        assert_eq!(loaded.len(), 3);
        assert_eq!(loaded.dimension(), 2);
        assert_eq!(
            loaded
                .select(&[4.0, 4.0], Metric::L2, NonZeroUsize::new(2).unwrap())
                .unwrap(),
            vec![20, 10]
        );
        assert!(storage.exists("/routing.bin"));
    }

    #[test]
    fn selection_deduplicates_node_ids() {
        let table = RoutingTable::new(vec![10, 10, 20], vec![0.0f32, 0.1, 1.0], 1).unwrap();
        assert_eq!(
            table
                .select(&[0.0], Metric::L2, NonZeroUsize::new(2).unwrap())
                .unwrap(),
            vec![10, 20]
        );
    }

    #[test]
    fn selection_prefers_largest_inner_product() {
        let table =
            RoutingTable::new(vec![10, 20, 30], vec![1.0f32, 0.0, 0.0, 1.0, 2.0, 2.0], 2).unwrap();
        assert_eq!(
            table
                .select(&[1.0, 1.0], Metric::InnerProduct, NonZeroUsize::MIN)
                .unwrap(),
            vec![30]
        );
    }
}
