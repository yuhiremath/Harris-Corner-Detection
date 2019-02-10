# Harris-Corner-Detection
Implementation of Corner detection using opencv

## Function (extract_keypoints)
### <i> Input : String (Image Path) </i>
### <i> Output : The coordinates of significant corners and the Cornerness score (X, Y, R[X,Y]) </i>
<br/>

### Setting the parameters
* The value of k is set. It will be used in the future code to calculate the cornerness score.
* Window size is defined.

### Read Image
* Image is read using openCV in grayscale.

### Calculating Image gradients
* I have used Sobel Edge detector for this task as it gives better overall result. 

### Calculating cornerness score
* For each neighbor in a 5x5 window calculate M[0,0], M[0,1], M[1,0] and M[1,1].
* By singular value decomposition, we can get the value of the eigen values.
* Eigen values can be used for the calculation of trace and determinant.
* Trace and determinant are used to calculate cornerness score(R).
* <i> The theory related to the above calculation can be found in this page --> https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_features_harris/py_features_harris.html </i>

### Cornerness score is thresholded
* Here the R(Cornerness Score) values above 5 times of mean of R are considered.
* Alternatively, different types of thresholding can be used.

### Performing Non-Maximum Suppression
* If cornerness score of a pixel is lower than all 8 of it's neighbors, then it is set to 0.

## Main method
* For this example I am using the an image of a panda.
* I also plotted the edges using circles with their radius depending on the cornerness score.

## Ouptut 
![The picture cannot be found](https://github.com/yuhiremath/Harris-Corner-Detection/blob/master/panda_corners.png)
