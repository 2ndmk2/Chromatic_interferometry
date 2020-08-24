cd imaging_color_source/tests
python main_sparse_mfreq_rings.py
cd ../../simulator_casa
python modified.py
casa -c sim_obs_clean.py
python make_figs.py
python make_compfigs.py
python folders_move.py
cd ../imaging_color_source/tests
python main_sparse_data_solve.py

cd ../../make_figures

python make_comp_clean_sp_figs.py