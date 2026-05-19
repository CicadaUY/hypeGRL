"""
Buffers for streaming graph updates.

EdgeBuffer
----------
Accumulates arriving/departing edges and revealed edge weights,
flushing to the embedder's update() method when a batch threshold
or time limit is reached.

NodeBuffer
----------
Accumulates node additions and deletions. Node additions are held
until their connecting edges are known (or declared unknown), then
flushed as a single update() call.
"""
# TODO: implement
