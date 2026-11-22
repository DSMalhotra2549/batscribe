# A GPU error stops the Batch; any other failure skips one Video

Most failures are specific to one Video (bad audio, unreadable file), so they are logged to the Error log and the Batch continues. A GPU error (message contains `CUDA`, `cuBLAS`, `cuDNN` or `out of memory`) would make every remaining Video fail too, flooding the logs and marking good Videos as failed, so it stops the Batch and tells the user to free the GPU and start again. Detection is by message text because CTranslate2 raises plain `RuntimeError`s.
