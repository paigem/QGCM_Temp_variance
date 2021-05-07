# QGCM_Temp_variance

This repo contains code used to compute the frequency-domain temperature variance budget equations in the idealized Quasi-Geostrophic Coupled Model (Q-GCM).

The primary workflow can be found in the folder `code_workflow/` and contains the following scripts/notebooks:
- `raijin_prep_data_gadi.py`: transpose and chunk the model output to make frequency-domain analysis faster (allows to access a full timeseries for spatial chunks without running into i/o problems)
- `QGCM_Tvar_funcs_28Nov2019.py`: contains functions that compute each of the terms in the frequency-domain temperature variance budget
- `QGCM_calc_Tvar_oc_30Jan2020.py`: takes user inputs to call the specified functions in `QGCM_Tvar_funcs_28Nov2019.py` to compute the terms
- `Plot_oc_Tvar_14Oct2020_forPaperReview_clean.ipynb`: plot the spatially integrated terms as a function of frequency
