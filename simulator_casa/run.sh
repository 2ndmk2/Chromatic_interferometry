FLAG=2

if [ $FLAG -lt 1 ]; then
	cd ../imaging_color_source/tests
	python main_sparse_mfreq_rings.py
	cd ../../simulator_casa
fi
if [ $FLAG -lt 2 ]; then
	python modified.py
	casa -c sim_obs_clean.py
	python make_figs.py
	python make_compfigs.py
fi

if [ $FLAG -lt 3 ]; then
	python folders_move.py
fi

