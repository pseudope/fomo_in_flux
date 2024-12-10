## Downloading LAION400M subset used in our experiments

### Download pool
Since the original LAION400M dataset is unavailable due to [significant ethical issues](https://purl.stanford.edu/kh752sm9123), you can reproduce all the results with the newly cleaned and curated [Re-LAION400M](https://laion.ai/blog/relaion-5b/) dataset. For example, a good dataset split to use would be the [Re-LAION2B-en](https://huggingface.co/datasets/laion/relaion2B-en-research). Please follow the guidelines using [img2dataset](https://github.com/rom1504/img2dataset/tree/main) to download the shards correctly. Note that you need not have the full dataset but a subset containing about ~1.3-1.5M image-text pairs should suffice. An easy to follow tutorial is available [here](https://github.com/rom1504/img2dataset/blob/main/dataset_examples/laion400m.md).

### Download statistics
Ensure that the downloaded subset roughly contains about ~1.3-1.5M image-text pairs. Our downloaded split on disk has about ~1.38M image-text pairs, however the exact numbers are not necessary for reproducibility.
