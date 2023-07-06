# TreeVision: A software tool for extracting tree biophysical parameters of trees from stereoscopic images

## About
Forest cover reduction poses a threat to environmental and ecological sustainability in Kenya. Various efforts to combat it, including reafforestation, face numerous challenges including prominently, lack of efficient, quick and rapid monitoring to inform timely policy changes by concerned stakeholders. We aim to explore various technologies that can be used to improve forest monitoring.

Stereoscopic vision is a computer vision technique that can be used retrieve the 3D information of a scene based on the concept of Multiple-View Geometry. *TreeVision* was developed to extend the principles of Multiple-View Geometry to facilitate fast and accurate estimation of tree attributes. 

*TreeVision* can accurately estimate the values of the diameter at breast height (DBH), crown diameter (CD), and tree height (CD). The DBH is usually measured at 1.3 m above the trunk base and the algorithms presented in *TreeVision* can estimate this location with impressive accuracy.

## Dataset
*TreeVision* has been tested and validated in a real forest setting. We have published this dataset on Mendeley Data and it is publicly available for use. The data set is titled [Tree Image Dataset for Biophysical Parameter Estimation using Stereoscopic Vision](https://www.doi.org/10.17632/nx3ggv7pxf.4). Visit this link and download the zip file called `tree_stereo_image_dataset_v4.zip`. 

## Test Usage
### Setting Up
1. Clone this repository. The root directory will be saved in your computer as `TreeVision`
```bash
git clone https://github.com/DeKUT-DSAIL/TreeVision.git
```

2. Download our dataset titled: [Tree Image Dataset for Biophysical Parameter Estimation using Stereoscopic Vision](https://www.doi.org/10.17632/nx3ggv7pxf.4)

3. Extract all the files from the zipped folder. You will have 4 folders (`full_trees`, `trunks`, `calib_v1`, `calib_v2`) and 2 files (`full_trees.csv`, `trunks.csv`). 

4. Transfer the `full_trees` and `trunks` folders to the root folder of the cloned repository (i.e., the `TreeVision` folder).

5. Create your own Python virtual environment usin either [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands) or [venv](https://docs.python.org/3/library/venv.html). This app was created using Python 3.9 but installing Python 3.6 and above in your virtual environment should work. Here is an example of how to create a conda environment with Python 3.9:

```bash
conda create -n treevision python=3.9
```

and here is an example using venv and python 3.9:

```bash
python3.9 -m venv [treevision]
```

7. Activate your python environment.

8. Install the packages in the file `requirements.txt`. Run this command:

```bash
pip install -r requirements.txt
```

If you run into trouble installing `kivy` package, read about its installation [here](https://kivy.org/doc/stable/gettingstarted/installation.html).

### Running the Application

1. Run the following command to start the application
```bash
python main.py
```