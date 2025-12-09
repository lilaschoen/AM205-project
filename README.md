# **AM205 Project — Low-Rank Image Approximation**

*Lila Schoen & Anastasia Ahani*

This repository contains MATLAB code for comparing truncated SVD, nuclear-norm minimization, and low-rank reconstruction methods.


## **Files**

* **compare_images.m** — Creates **Figure 2**; compares singular-value decay and energy capture across images.

* **compare_svd_nuclear_norm_image_visualization.m** — Creates **Figure 3**; visual side-by-side SVD vs nuclear-norm approximations at fixed ranks.

* **compare_svd_nuclear_norm_graphs.m** — Creates **Figure 4**; plots error vs rank and error vs runtime for SVD vs nuclear norm.

* **comparison_image_recomposition.m** — Creates **Figures 5–8**; compares reconstruction methods on masked images.

* **run_nuclear_norm_function.m** — Wrapper to run nuclear-norm (SVT) low-rank approximation.

* **nuclear_norm_svt.m** — Core SVT algorithm for nuclear-norm minimization.

* **run_truncated_svd_function.m** — Runs truncated SVD for low-rank approximation.

* **image_recomposition_low_rank_factorization_function.m** — Low-rank factorization method for image completion.

* **image_recomposition_naive_function.m** — Naive baseline reconstruction.
