"""
- Baseline
Examples/sec (First excluded) 628.03 ex/sec (total: 256512 ex, 408.44 sec)
- num_parallel_calls_for_interleave_files=1
Examples/sec (First excluded) 578.85 ex/sec (total: 256512 ex, 443.14 sec)
- num_parallel_calls_for_interleave_files=4
Examples/sec (First excluded) 625.96 ex/sec (total: 256512 ex, 409.79 sec)
- num_parallel_calls=16:
Examples/sec (First excluded) 622.29 ex/sec (total: 256512 ex, 412.21 sec)
- Shuffle-batch-repeat:
Examples/sec (First excluded) 624.53 ex/sec (total: 256512 ex, 410.73 sec)
- prefetch_buffer_size=16K
Examples/sec (First excluded) 632.56 ex/sec (total: 256512 ex, 405.52 sec)
- num_parllel_calls=128:
Examples/sec (First excluded) 734.43 ex/sec (total: 256512 ex, 349.27 sec)
- prefetch_buffer_size=32K, num_parallel_calls=256
Examples/sec (First excluded) 980.06 ex/sec (total: 256512 ex, 261.73 sec)
- prefetch_buffer_size=128K, num_parallel_calls=1024:
Examples/sec (First excluded) 1265.84 ex/sec (total: 256512 ex, 202.64 sec)
- prefetch_buffer_size=128 * 1024, read_parallelism=1024, process_parallelism=64:
Examples/sec (First excluded) 646.21 ex/sec (total: 256512 ex, 396.95 sec)
- prefetch_buffer_size=128 * 1024, read_parallelism=64, process_parallelism=1024:
Examples/sec (First excluded) 1290.59 ex/sec (total: 256512 ex, 198.76 sec)
- prefetch_buffer_size=128 * 1024, read_parallelism=64, process_parallelism=512:
Examples/sec (First excluded) 1188.40 ex/sec (total: 256512 ex, 215.85 sec)
- prefetch_buffer_size=128 * 1024, read_parallelism=64, process_parallelism=256:
Examples/sec (First excluded) 988.22 ex/sec (total: 256512 ex, 259.57 sec)
- read_parallelism, process_parallelism = 64, 1024, prefetch_buffer_size=read_parallelism * 1024:
Examples/sec (First excluded) 1067.68 ex/sec (total: 256512 ex, 240.25 sec) (num_iters=1000)
Examples/sec (First excluded) 1080.92 ex/sec (total: 128512 ex, 118.89 sec) (num_iters=500)
Examples/sec (First excluded) 1085.29 ex/sec (total: 128512 ex, 118.41 sec)
- read_parallelism, process_parallelism = 128, 1024, prefetch_buffer_size=read_parallelism * 1024:
Examples/sec (First excluded) 736.54 ex/sec (total: 128512 ex, 174.48 sec)
- read_parallelism, process_parallelism = 32, 1024
Examples/sec (First excluded) 1118.65 ex/sec (total: 128512 ex, 114.88 sec)
Examples/sec (First excluded) 1163.83 ex/sec (total: 128512 ex, 110.42 sec)
- read_parallelism, process_parallelism = 16, 1024
Examples/sec (First excluded) 1224.25 ex/sec (total: 128512 ex, 104.97 sec)
- read_parallelism, process_parallelism = 8, 1024
Examples/sec (First excluded) 1260.53 ex/sec (total: 128512 ex, 101.95 sec)
- read_parallelism, process_parallelism = 4, 1024
Examples/sec (First excluded) 1268.11 ex/sec (total: 128512 ex, 101.34 sec)
- read_parallelism, process_parallelism = 2, 1024
Examples/sec (First excluded) 1271.49 ex/sec (total: 128512 ex, 101.07 sec)
- read_parallelism, process_parallelism = 1, 1024
Examples/sec (First excluded) 1267.19 ex/sec (total: 128512 ex, 101.41 sec)
- read_parallelism, decode_parallelism, process_parallelism = 1, 1, 1024
Examples/sec (First excluded) 367.91 ex/sec (total: 128512 ex, 349.30 sec)
- read_parallelism, decode_parallelism, process_parallelism = 1, 32, 1024
Examples/sec (First excluded) 1104.97 ex/sec (total: 128512 ex, 116.30 sec)
- read_parallelism, decode_parallelism, process_parallelism = 1, 128, 1024
Examples/sec (First excluded) 1291.70 ex/sec (total: 128512 ex, 99.49 sec)
Examples/sec (First excluded) 1207.42 ex/sec (total: 128512 ex, 106.44 sec)
- read_parallelism, decode_parallelism, process_parallelism = 1, 128, 128
Examples/sec (First excluded) 778.76 ex/sec (total: 128512 ex, 165.02 sec)
- read_parallelism, decode_parallelism, process_parallelism = 1, 128, 512
Examples/sec (First excluded) 1157.43 ex/sec (total: 128512 ex, 111.03 sec)
"""

from image import ImagenetInput
import tensorflow_datasets as tfds

read_parallelism, decode_parallelism, process_parallelism = 1, 128, 1024
cfg = ImagenetInput.default_config().set(
    name="benchmark", split="train", is_training=True,
    prefetch_buffer_size=read_parallelism * 1024,
    shuffle_buffer_size=read_parallelism * 1024,
    read_parallelism=read_parallelism,
    decode_parallelism=decode_parallelism,
    process_parallelism=process_parallelism,
    global_batch_size=256, data_dir="gs://permanent-us-central1-q5loch/tensorflow_datasets")
inputs = cfg.instantiate(parent=None)
print(tfds.benchmark(inputs, batch_size=256, num_iter=500).stats)
