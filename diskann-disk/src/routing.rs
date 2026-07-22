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
    storage::{StorageReadProvider, StorageWriteProvider},
    utils::{gen_random_slice, RayonThreadPoolRef, VectorDataIterator},
};
use diskann_vector::{distance::Metric, DistanceFunction};
use rand::Rng;

use crate::utils::{compute_closest_centers, k_means_clustering};

const ROUTING_TABLE_MAGIC: &[u8; 8] = b"DANNRTE1";
const REPRESENTATIVE_BATCH_SIZE: usize = 1_200;

/// In-memory vectors and graph IDs used to choose query-specific graph entry points.
#[derive(Debug, Clone)]
pub struct RoutingTable<T> {
    ids: Vec<u32>,
    vectors: Vec<T>,
    dimension: usize,
}

impl<T> RoutingTable<T>
where
    T: VectorRepr,
{
    /// Construct a routing table from real graph node IDs and vectors.
    pub fn new(ids: Vec<u32>, vectors: Vec<T>, dimension: usize) -> ANNResult<Self> {
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
    pub fn select(&self, query: &[T], metric: Metric, count: NonZeroUsize) -> ANNResult<Vec<u32>> {
        if query.len() != self.dimension {
            return Err(ANNError::log_index_error(format_args!(
                "query dimension {} does not match routing dimension {}",
                query.len(),
                self.dimension
            )));
        }

        let distance = T::distance(metric, Some(self.dimension));
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
        if element_size != std::mem::size_of::<T>() {
            return Err(ANNError::log_invalid_file_format(format!(
                "routing table element size {} does not match requested type size {}",
                element_size,
                std::mem::size_of::<T>()
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
        let mut vector = vec![T::default(); dimension];
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
        writer.write_u32::<LittleEndian>(std::mem::size_of::<T>() as u32)?;
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
) -> ANNResult<RoutingTable<T>>
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
    let stored_dimension = dataset.get_dimension();
    let num_points = dataset.get_num_points();
    let mut best_distances = vec![f32::INFINITY; num_centers.get()];
    let mut representative_ids = vec![0; num_centers.get()];
    let mut representative_vectors = vec![T::default(); num_centers.get() * stored_dimension];
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

        for (row, ((vector, ()), center_id)) in batch.iter().zip(closest_centers).enumerate() {
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
                    [center_id * stored_dimension..(center_id + 1) * stored_dimension]
                    .copy_from_slice(vector);
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

    let table = RoutingTable::new(representative_ids, representative_vectors, stored_dimension)?;
    table.save(output_path, storage_provider)?;
    Ok(table)
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;

    use diskann_providers::storage::{StorageReadProvider, VirtualStorageProvider};
    use diskann_vector::distance::Metric;
    use vfs::MemoryFS;

    use super::*;

    #[test]
    fn routing_table_round_trip_and_selection() {
        let storage = VirtualStorageProvider::new(MemoryFS::new());
        let table =
            RoutingTable::new(vec![10, 20, 30], vec![0.0f32, 0.0, 5.0, 5.0, 9.0, 9.0], 2).unwrap();
        table.save("/routing.bin", &storage).unwrap();

        let loaded = RoutingTable::<f32>::load("/routing.bin", &storage).unwrap();
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
}
