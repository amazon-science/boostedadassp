# Boosted AdaSSP
This repo contains code pieces for our algorithm --- [Boosted AdaSSP](https://openreview.net/pdf?id=LrAAGxe8HY). It is a differentially private linear regression model that uses gradient boosting on top of AdaSSP to improve its performance.
## Requiements
- numpy
- scipy
- scikit-learn(>=1.2)
- pandas
- openml

## How to run
We provide two python scripts for running classification and regression tasks, respectively, using OpenML tasks.
```shellscript
python run_regression.py
```
## License

This library is licensed under the Apache-2.0 License. See the LICENSE file.

## Citation
Please use the following citation when publishing material that uses our code:
```tex
@article{tang2023improved,
  title={Improved Differentially Private Regression via Gradient Boosting},
  author={Tang, Shuai and Aydore, Sergul and Kearns, Michael and Rho, Saeyoung and Roth, Aaron and Wang, Yichen and Wang, Yu-Xiang and Wu, Zhiwei Steven},
  journal={2nd IEEE Conference on Secure and Trustworthy Machine Learning},
  year={2024}
}
```
