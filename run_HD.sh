FLAG=0

## Making input images
## if FLAG < 1
if [ $FLAG -lt 1 ]; then

	cd imaging_color_source/tests
	python main_sparse_mfreq_HD14like.py
	cd ../../
fi


## Making ALMA Observations and CLEANed images

## if FLAG < 2
if [ $FLAG -lt 2 ]; then
	cd simulator_casa
	python modified.py
	casa -c sim_obs_clean.py
	python after_clean.py
	cd ..
fi

## Sparse imaging
## if FLAG < 3
if [ $FLAG -lt 3 ]; then
	cd imaging_color_source/tests
	python main_sparse_data_solve.py
	cd ../../make_figures
	python make_comp_clean_sp_figs_2.py
fi