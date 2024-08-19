DST_RVL_CDIP
==============================

installation du projet
------------

python3 -m venv env_RVL_CDIP

source env_RVL_CDIP/bin/activate

pip install -r requirements.txt

Description des imports
------------
pandas numpy

scikit-learn

matplotlib

seaborn


Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data               <- Should be in your computer but not on Github (only in .gitignore)
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's name, and a short `-` delimited description, e.g.
    │                         `1.0-alban-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, links, and all other explanatory materials.
    │
    ├── reports            <- The reports that you'll make during this project as PDF
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   ├── visualization  <- Scripts to create exploratory and results oriented visualizations
    │   │   └── visualize.py

Description des datas
------------
lien des datas : https://adamharley.com/rvl-cdip/

projets existants sur ces datas : https://paperswithcode.com/sota/document-image-classification-on-rvl-cdip

![RVL_CDIP](https://github.com/user-attachments/assets/c6b260cf-418d-4f9d-9ba8-ffac4b8f37b4)

________________

RVL-CDIP Dataset
________________

The RVL-CDIP (Ryerson Vision Lab Complex Document Information Processing) dataset consists of 400,000 grayscale images in 16 classes, with 25,000 images per class. There are 320,000 training images, 40,000 validation images, and 40,000 test images. The images are sized so their largest dimension does not exceed 1000 pixels.

For questions and comments please contact Adam Harley (aharley@scs.ryerson.ca).

_________

CHANGELOG
_________

05/JUN/2015	First version of the dataset

_______

DETAILS
_______

The label files list the images and their categories in the following format:

path/to/the/image.tif category

where the categories are numbered 0 to 15, in the following order:

0 letter
1 form
2 email
3 handwritten
4 advertisement
5 scientific report
6 scientific publication
7 specification
8 file folder
9 news article
10 budget
11 invoice
12 presentation
13 questionnaire
14 resume
15 memo

________

CITATION
________

If you use this dataset, please cite:

A. W. Harley, A. Ufkes, K. G. Derpanis, "Evaluation of Deep Convolutional Nets for Document Image Classification and Retrieval," in ICDAR, 2015

Bibtex format:

@inproceedings{harley2015icdar,
    title = {Evaluation of Deep Convolutional Nets for Document Image Classification and Retrieval},
    author = {Adam W Harley and Alex Ufkes and Konstantinos G Derpanis},
    booktitle = {International Conference on Document Analysis and Recognition ({ICDAR})}},
    year = {2015}
}

___________________

FURTHER INFORMATION
___________________

This dataset is a subset of the IIT-CDIP Test Collection 1.0 [1]. The file structure of this dataset is the same as in the IIT collection, so it is possible to refer to that dataset for OCR and additional metadata. The IIT-CDIP dataset is itself a subset of the Legacy Tobacco Document Library [2].

[1] D. Lewis, G. Agam, S. Argamon, O. Frieder, D. Grossman, and J. Heard, "Building a test collection for complex document information processing," in Proc. 29th Annual Int. ACM SIGIR Conference (SIGIR 2006), pp. 665-666, 2006
[2] The Legacy Tobacco Document Library (LTDL), University of California, San Francisco, 2007. http://legacy.library.ucsf.edu/.

More information about this dataset can be obtained at the following URL: http://scs.ryerson.ca/~aharley/rvl-cdip/


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
