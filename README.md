# TreeVision: A software tool for extracting tree biophysical parameters of trees from stereoscopic images

## About
Forest cover reduction poses a threat to environmental and ecological sustainability in Kenya. Various efforts to combat it, including reafforestation, face numerous challenges including prominently, lack of efficient, quick and rapid monitoring to inform timely policy changes by concerned stakeholders. We aim to explore various technologies that can be used to improve forest monitoring.

Stereoscopic vision is a computer vision technique that can be used retrieve the 3D information of a scene based on the concept of Multiple-View Geometry. *TreeVision* was developed to extend the principles of Multiple-View Geometry to facilitate fast and accurate estimation of tree attributes. 

*TreeVision* can accurately estimate the values of the diameter at breast height (DBH), crown diameter (CD), and tree height (CD). The DBH is usually measured at 1.3 m above the trunk base and the algorithms presented in *TreeVision* can estimate this location with impressive accuracy. The algorithms for estimating these parameters are found inside the [Controller/algorithms.py](./Controller/algorithms.py) file

*TreeVision* is built using the [Kivy](https://kivy.org/) and [KivyMD](https://kivymd.readthedocs.io/en/1.1.1/) Python packages.

## Dataset
*TreeVision* has been tested and validated on with a dataset obtained from a real forest setting. We have published this dataset on Mendeley Data and it is publicly available for use. The data set is titled [Tree Image Dataset for Biophysical Parameter Estimation using Stereoscopic Vision](https://www.doi.org/10.17632/nx3ggv7pxf.4). Visit this link and download the zip file called `tree_stereo_image_dataset_v4.zip`. 

## Test Usage
### Setting Up
1 Clone this repository. The root directory will be saved in your computer as `TreeVision`
```bash
git clone https://github.com/DeKUT-DSAIL/TreeVision.git
```

Download our dataset titled [Tree Image Dataset for Biophysical Parameter Estimation using Stereoscopic Vision](https://www.doi.org/10.17632/nx3ggv7pxf.4) from Mendeley Data.

3 Extract all the files from the zipped folder. You will have 4 folders (`full_trees`, `trunks`, `calib_v1`, `calib_v2`) and 2 files (`full_trees.csv`, `trunks.csv`). 

4 Transfer the `full_trees` and `trunks` folders to the root folder of the cloned repository (i.e., the `TreeVision` folder).

5 Open the application's root folder (`TreeVision`) in a terminal application. Use `bash` if you are on Linux or `Git Bash` if you are on Windows. Using `Git Bash` on Windows will make it possible to finish your setup process using the commands in steps 6 and 7.

6 From the application's too directory, run the `setup.sh` script using the command:
```bash
source setup.sh
```

7 Run the `install.sh` script using the command:
```bash
source install.sh
```

### Test Parameter Extraction
A folder called `test` is included in the repository. It contains a test set of 20 image pairs of full trees from which you can extract CDs and THs.Follow these steps:

1 On the user interface, click on the green `Extract` button and watch what happens on the widget at the bottom right of the application. Some information on the extracted CD and Th of the tree displayed on the UI has now been displayed.

2 On the user interface, click on the green `Batch Extract` button. The app will now batch extract the parameters of all 20 trees in the `test` folder and display them on the widget at the bottom right corner. The batch extraction will be complete once the UI is no longer being updated with new images.

3 In your file explorer, navigate to the [test folder](./assets/projects/test/results/results.csv) to find a `results.csv` file and open it. This file contains the extracted parameters of all the 20 trees. You can open it using MS Excel for a better view.



# Congratulations !!!
You have successfully extracted tree parameters of 20 trees using *TreeVision*.