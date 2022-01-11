### download data

	conda activate pytorch
	cd S:\Projects\DGLVM\DataSet

# Area2_Bump: 
	dandi download https://dandiarchive.org/dandiset/000127
# MC_Maze:
	dandi download https://dandiarchive.org/dandiset/000128
# MC_RTT: 
	dandi download https://dandiarchive.org/dandiset/000129
# DMFC_RSG: 
	dandi download https://dandiarchive.org/dandiset/000130

# MC_Maze_Large: 
	dandi download https://dandiarchive.org/dandiset/000138
# MC_Maze_Medium: 
	dandi download https://dandiarchive.org/dandiset/000139
# MC_Maze_Small: 
	dandi download https://dandiarchive.org/dandiset/000140


### Run model

	export CUDA_VISIBLE_DEVICES=1

	source ~/.bashrc
	source activate tensorflow
	cd ~/Projects/NFLVM
	nvidia-smi
	python Run.py --model NFLVM --times 8
	
	python Run.py --model SNLVM --times 1
	python Run.py --model LFADS --times 1
	python Run.py --model TNDM  --times 1

	nohup python Run.py --model SNLVM --times 4 > ./SNLVM.log &
	nohup python Run.py --model LFADS --times 4 > ./LFADS.log &
	nohup python Run.py --model TNDM  --times 4 > ./TNDM.log &
