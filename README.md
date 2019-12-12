# MS-RCAE-for-HSIC
A superpixel-based relational auto-encoder for feature extraction of hyperspectral images

https://www.mdpi.com/2072-4292/11/20/2454

# Getting starded

After downloading the code, firstly down load the pretrained VGG-16 'imagenet-vgg-verydeep-16.m' from 'http://www.vlfeat.org/matconvnet/pretrained/', and put it into the folder 'data/models';

run the 'multiscale_for_spatial.m' and 'multiscale_for_spectral.m'file to extract deep spatial features and corresponding spectral features;  

Then, run the 'GCSAE_Multiscale_demo.m'file for multiscale spectral-spatial feature fusion;

Finally, run the 'GCSAE_SVMLinear_demo.m' file for HSI classification.

Besides, code 'Trainsamples_demo.m' for randomly selecting train and test samples.

# Citation

If you use the code, please kindly cite this paper "Miaomiao Liang, Licheng Jiao, Zhe Meng."A Superpixel-Based Relational Auto-Encoder for Feature Extraction of Hyperspectral Images," Remote Sensing, vol. 11, no. 20, pp. 2454, Oct. 2019."

@article{liang2019superpixel,  
  title={A Superpixel-Based Relational Auto-Encoder for Feature Extraction of Hyperspectral Images},  
  author={Liang, Miaomiao and Jiao, Licheng and Meng, Zhe},  
  journal={Remote Sensing},  
  volume={11},  
  number={20},  
  pages={2454},  
  year={2019},  
  publisher={Multidisciplinary Digital Publishing Institute}  
}

If you have any questions or suggestions, welcome to contact me by email:liangmiaom@gmail.com
