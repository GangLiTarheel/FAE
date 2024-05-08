# DFI: FAE
### Deep fiducial inference

Since the mid‐2000s, there has been a resurrection of interest in modern modifications of fiducial inference. To date, the main computational tool to extract a generalized fiducial distribution is Markov chain Monte Carlo (MCMC). We propose an alternative way of computing a generalized fiducial distribution that could be used in complex situations. In particular, to overcome the difficulty when the unnormalized fiducial density (needed for MCMC) is intractable, we design a fiducial autoencoder (FAE). The fitted FAE is used to generate generalized fiducial samples of the unknown parameters. To increase accuracy, we then apply an approximate fiducial computation (AFC) algorithm, by rejecting samples that when plugged into a decoder do not replicate the observed data well enough. Our numerical experiments show the effectiveness of our FAE‐based inverse solution and the excellent coverage performance of the AFC‐corrected FAE solution.

## News and Updates
Dec 21, 2020
* First release

## Required Packages
Python 3.6.10
Keras 2.2.4
  
## Citation
@article{li2020deep,
  title={Deep fiducial inference},
  author={Li, Gang and Hannig, Jan},
  journal={Stat},
  volume={9},
  number={1},
  pages={e308},
  year={2020},
  publisher={Wiley Online Library}
}

Li, G, Hannig, J. Deep fiducial inference. Stat. 2020; 9e308. https://doi.org/10.1002/sta4.308

https://onlinelibrary.wiley.com/doi/abs/10.1002/sta4.308

Contact: Gang Li [gangliuw@uw.edu].
