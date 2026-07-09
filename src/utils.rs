use itertools::Itertools;
use rayon::prelude::*;

struct SendPtr<T>(*mut T);

unsafe impl<T> Sync for SendPtr<T> {}
// unsafe impl<T> Send for SendPtr<T> {}

pub fn parallel_concat<T: Send + Sync>(arrs: &[impl AsRef<[T]> + Send + Sync]) -> Vec<T> {
    let lens = arrs
        .iter()
        .map(|e| e.as_ref().len())
        .collect::<Vec<usize>>();
    let start_idcs = lens
        .iter()
        .scan(0, |acc, &x| {
            let old = *acc;
            *acc += x;
            Some(old)
        })
        .collect::<Vec<usize>>();

    let total_len = arrs.iter().map(|e| e.as_ref().len()).sum();
    let mut result = Vec::with_capacity(total_len);

    let result_ptr = SendPtr(result.as_mut_ptr());

    unsafe {
        result.set_len(total_len);
    }

    arrs.par_iter()
        .zip(start_idcs)
        .for_each(|(arr, start_idx)| unsafe {
            let sent = &result_ptr;
            let out_ptr: *mut T = sent.0;
            let result_ptr = out_ptr.add(start_idx);
            std::ptr::copy_nonoverlapping(arr.as_ref().as_ptr(), result_ptr, arr.as_ref().len())
        });

    result
}

/// Return num_chunks + 1 boundary byte indices that split the bytes into num_chunks roughly equal chunks while avoiding splitting a character in the middle of a multi-byte UTF-8 sequence.
pub fn chunks_at_utf8_boundaries(bytes: &[u8], num_chunks: usize) -> Vec<usize> {
    if num_chunks == 0 {
        return vec![0, bytes.len()];
    }
    let len = bytes.len();
    // Compute the size of each chunk (possibly uneven division)
    let chunk_size = len.div_ceil(num_chunks);

    let mut boundaries = Vec::with_capacity(num_chunks + 1);
    boundaries.push(0);

    let mut next = chunk_size;

    for i in 1..num_chunks {
        let mut boundary = next;
        // Move forward to the next valid UTF-8 character boundary
        // (Never go beyond the end of bytes)
        while boundary < len && (bytes[boundary] & 0b1100_0000) == 0b1000_0000 {
            boundary += 1;
        }
        boundaries.push(boundary);
        next += chunk_size;
    }
    // Last boundary is at the end
    boundaries.push(len);

    boundaries
}
