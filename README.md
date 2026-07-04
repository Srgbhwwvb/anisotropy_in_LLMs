# anisotropy_in_LLMs
Code for the paper "Periodic Nature of Anisotropy in Transformers: How Components of Transformer Layers Affect the Geometry of Vectors of Token Representations" (accepted at the AINL Conference, 2026).
Link: https://drive.google.com/file/d/1n_3uciOl4dKd8Q6GiL1WzTk6PfArEFel/view?usp=sharing

For token representations in each LLM layer and after each transformation (attention, MLP, normalization, residuals), the code computes:
- cosine anisotropy,
- singular anisotropy,
- effective dimensions,
- intrinsic dimensions.
Also, code allows to monitor these characteristics over finetuning and modified architectures.

We found a consistent periodic pattern across three model families (Pythia, Qwen, GPT2), two datasets (Wikipedia, Realnewslike), and all of the above characteristics. Namely, the values spike after residual connections but drop after attention and MLP blocks.
