
## GOAS Source Code

Provided is code that demonstrates the training and evaluation of the work presented in the paper: "Noise Modeling, Synthesis, and Classification for Generic Object Anti-Spoofing" published in CVPR 2020.

![The proposed training framework with noise modeling](https://github.com/JStehouwer/GOAS_CVPR2020/blob/master/readme_fig.png)

### Project Webpage

See the MSU CVLab website for project details and access to the GOSet dataset.

http://cvlab.msu.edu/project-goas.html

### Notes

This code is provided as example code, and may not reflect a specific combination of hyper-parameters presented in the paper.

### Description of files

- `prepare_dats.py`: Processes the dataset into binary files for network training
- `database.py`: Reads prepared .dat files during network training
- `networks.py`: Defines the structure and operations of the networks
- `golab_train.py`: Trains the GOLab network
- `golab_freeze.py`: Optimizes and freezes the GOLab model for evaluation
- `golab_eval.py`: Evaluates the frozen GOLab model
- `golab_perf.m`: Compute evaluation metrics for GOLab
- `gogen_train.py`: Trains the GOGen network
- `gogen_freeze.py`: Optimizes and freezes the GOGen model for evaluation
- `gogen_eval.py`: Evaluates the frozen GOGen model
- `gogen_perf.m`: Compute evaluation metrics for GOGen

### Acknowledgements

If you use or refer to this source code, please cite the following paper:

	@inproceedings{cvpr2020-stehouwer,
	  title={Noise Modeling, Synthesis, and Classification for Generic Object Anti-Spoofing},
	  author={Joel Stehouwer, Amin Jourabloo, Yaojie Liu, Xiaoming Liu},
	  booktitle={In Proceeding of IEEE Computer Vision and Pattern Recognition (CVPR 2020)},
	  address={Seattle, WA},
	  year={2020}
	}

