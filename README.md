# anisotropy_in_LLMs
Code for paper "Periodic nature of anisotropy in transformers: how components of transformer layers affect the geometry of vectors of token representations" (accepted in AINL Conference, 2026) https://drive.google.com/file/d/1n_3uciOl4dKd8Q6GiL1WzTk6PfArEFel/view?usp=sharing

For token representations in each LLM's layer and after each transformation (attention, MLP, normalization, residuals) code compute:
- cosine anisotropy
- singular anisotropy
- effective dimensions
- intrinsic dimensions
    
We found a consistent periodic pattern across three model families (Pythia, Qwen, GPT2), two datasets (Wikipedia, Realnewslike), and for all of the above characteristics,namely, values spike after residual connections but drop after attention and MLP blocks.
