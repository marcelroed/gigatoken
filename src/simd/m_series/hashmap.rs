//! HashMap implementations supporting SIMD operations

/*
Assume we can get a stream of byte chunks, as well as boundary indices for pretokens within each chunk.
We now want to use SIMD operations to create a rolling hash for the byte chunks, using the state from the previous chunk.
Whenever we encounter a boundary index, we want to update the hash with the new pretoken.
*/


struct BufferHashMap {
    local_buffer: [u8; 128],
    
}