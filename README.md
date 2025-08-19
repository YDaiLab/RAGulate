# RAGulate<img src="assets/RAGulate_logo.png" alt="RAGulate logo" align="right" width="200" />
**R**etrieval-**A**ugmented **G**eneration for Post-hoc Literature-Grounded **Regulatory** Validation

[![GitHub issues](https://img.shields.io/github/issues/YDaiLab/RAGulate)](https://github.com/YDaiLab/RAGulate/issues)
[![PyPI - Project](https://img.shields.io/pypi/v/ragulate-bio)](https://pypi.org/project/ragulate-bio/)
[![Conda](https://img.shields.io/conda/v/zandigohar/RAGulate?label=conda)](https://anaconda.org/zandigohar/RAGulate)
[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://ydailab.github.io/RAGulate/)


## Introduction
**RAGulate** is a **R**etrieval-**A**ugmented **G**eneration (RAG) pipeline that integrates domain-specific large language models (LLM) with curated literature to identify, score, and validate inferred **regulatory** (transcription factor-target gene) interactions in their biological context.

For further information and example tutorials, please check our documentation:

If you have any questions or concerns, feel free to [open an issue](https://github.com/YDaiLab/RAGulate/issues).

## Requirements
RAGulate is implemented in the LlamaIndex framework. Running RAGulate on `CUDA` is highly recommended if available.


Before installing and running RAGulate, ensure you have the following libraries installed:
- **PyTorch** (version 2.0 or higher)  
  Install with the exact command from the [PyTorch “Get Started” page](https://pytorch.org/get-started/locally/) for your OS, Python version and (optionally) CUDA toolkit.
- **NumPy** (version 1.23 or higher)

You can install these dependencies using `pip`:

```bash
pip install torch numpy
```

## Installation

**Option 1 (Coming soon):**  
You can install **RAGulate** via pip for a lightweight installation:

```bash
pip install ragulate-bio
```

**Option 2 (Coming soon):**  
Alternatively, if you want the latest, unreleased version, you can install it directly from the source on GitHub:

```bash
pip install git+https://github.com/YDaiLab/RAGulate.git
```

**Import**

```python
import ragulate_bio as ragulate  # recommended alias
```

Note: The PyPI distribution is named ragulate-bio to avoid a name conflict with an unrelated project called ragulate.
Always import ragulate_bio in Python (you may alias it to ragulate for convenience)

**Option 3 (Coming soon):**  
For users who prefer Conda or Mamba for environment management, you can install **RAGulate** along with extra dependencies:

**Conda:**
```bash
conda install -c zandigohar RAGulate
```

**Mamba:**
```bash
mamba create -n RAGulate -c zandigohar RAGulate
```

## FAQ

**Q1: Do I need a GPU to run RAGulate?**  
No, a GPU is not required. However, using a CUDA-enabled GPU is strongly recommended for faster runs, especially with large queries.

**Q2: How do I know if I can use a GPU with RAGulate?**  
There are two quick checks:

1. **System check**  
   In your terminal, run `nvidia-smi`. If you see your GPU listed (model, memory, driver version), your machine has an NVIDIA GPU with the driver installed.

2. **Python check**  
   In a Python shell, run:
   ```python
   import torch
   print(torch.cuda.is_available())  # True means PyTorch can see your GPU
   print(torch.cuda.device_count())  # How many GPUs are usable
   ```

**Q3: Can I use RAGulate with R-based tools?**  
RAGulate is written in Python and works directly with `Numpy` objects. 

**Q4: What if I also have another package called ragulate installed?**
RAGulate will warn you if it detects a conflicting installation. We recommend using a clean virtual environment to avoid import clashes.

**Q5: How do I cite RAGulate?**  
See the [Citation](#citation) section below for the latest reference and preprint link.

**Q6: How can I reproduce the paper’s results?**  
See our [Reproducibility Guide](https://github.com/YDaiLab/RAGulate/blob/main/notebooks/XYZ.ipynb) for step-by-step instructions. Then run RAGulate.

## Citation

⚠️ Preprint coming soon (bioRxiv, 2025)  
**This repository is under active development.** Please cite as:  
Zandigohar M, Rehman J, Dai Y. RAGulate: RAGulate: Retrieval-Augmented Generation for Post-hoc Literature-Grounded Regulatory Validation. 2025.


## Development & Contact
RAGulate was developed and is actively maintained by Mehrdad Zandigohar as part of his PhD research at the University of Illinois Chicago (UIC), in the lab of Dr. Yang Dai.

📬 For private questions, please email: mzandi2@uic.edu

🤝 For collaboration inquiries, please contact PI: Dr. Yang Dai (yangdai@uic.edu)

Contributions, feature suggestions, and feedback are always welcome!

## License

The code in **RAGulate** is licensed under the [MIT License](https://opensource.org/licenses/MIT), which permits academic and commercial use, modification, and distribution. 

Please note that any third-party dependencies bundled with **RAGulate** may have their own respective licenses.

