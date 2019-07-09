# tf_bees

Bee hive temperature denoising using RBF with TensorFlow

This python app receives as input a file containing bee hive temperatures contaminated with non-gaussian noise and produces an output file with the denoised data.

The denoising is based on an RBF (Radial Basis Function) neural network trained with Correntropy criterion. The software uses TensorFlow to generate the neural net model and training steps.

The input data is normalized so the output do not contain the actual temperature, the user have to renormalized the output after denoising. The normalization depends on the calibration of the sensor and the user should be aware of that fact.

## Usage 

./rbf.py inputfile column outputfile

The input file should be a text file with n columns of measurements. Each line is a measurement in time and each colum is a sensor. The column parameter is the number of the sensor (column of the file). utput file will be a text file with the denoised data.


Example:

./rbf.py bees2.txt 2 bees2_res_2.txt

