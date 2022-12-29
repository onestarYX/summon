# Scene Synthesis from Human Motion
![representative_img](https://user-images.githubusercontent.com/25496380/189529534-dcdd01f5-8422-410a-8de5-6a6404f81d37.png)

This is the official repository for the paper: **Scene Synthesis from Human Motion** [[Paper]()]

## Installation
### Environment
We highly recommand you to create a Conda environment to better manage all the Python packages needed.
```
conda create -n summon python=3.8
conda activate summon
```
After you create the environment, please install pytorch with CUDA. You can do it by running
```
conda install pytorch pytorch-cuda=11.6 -c pytorch -c nvidia
```
The other dependencies needed for this work is listed in the requirements.txt. 
We recommend you to use pip to install them: 
```
pip install -r requirements.txt
```

### Datasets and Model Checkpoints
We provide our preprocessed [PROXD](https://ps.is.mpg.de/uploads_file/attachment/attachment/530/ICCV_2019___PROX.pdf)
dataset and several testing motion sequences from the [AMASS](https://files.is.tue.mpg.de/black/papers/amass.pdf) 
dataset. Please use this [link](https://drive.google.com/file/d/1RcYoQMSqYUpVLEP45TqZO1ASPJkswsJr/view?usp=share_link)
to download. After downloading, please unzip it in the project root directory.

We provide a pretrained [model checkpoint](https://drive.google.com/file/d/1JZsRFCjUUEgHre8qtpu8v2bF0FwIb6hK/view?usp=sharing)
for ContactFormer. Please download and unzip it in the project root directory.
After doing that, you will get a `training/` folder with two subfolders: 
`contactformer/` and `posa/`.

We also provide a small subset of 3D_Future for you to test. Please use this [link](https://drive.google.com/file/d/10teWFzqB7Z_X-Om5rmCdbN_UVUEEP9_9/view?usp=share_link)
to download and unzip it at the root directory.

## Contact Prediction
To generate contact label predictions for all motion sequences stored in the 
`.npy` format in a folder (e.g. `amass/` in our provided data folder),
you can run
```
cd contactFormer
python predict_contact.py ../data/amass --load_model ../training/contactformer/model_ckpt/best_model_recon_acc.pt --output_dir PATH_OF_OUTPUT
```
Please replace the `PATH_OF_OUTPUT` to any path you want. If you want to
generate predictions for the PROXD dataset, you can try
```
python predict_contact.py ../data/proxd_valid/vertices_can --load_model ../training/contactformer/model_ckpt/best_model_recon_acc.pt --output_dir PATH_OF_OUTPUT
```
The above example command generate predictions for the validation split of PROXD.
If you want save the probability for each contact object category in order
to generate more diverse scenes, you can add a `--save_probability` flag
in addition to the above command.

### Train and test ContactFormer
You can train and test your own model. To train the model, still under `contactformer/`, you can run
```
python train_contactformer.py --train_data_dir ../data/proxd_train --valid_data_dir ../data/proxd_valid --fix_ori --epochs 1000 --out_dir ../training --experiment default_exp
```
Replace the train and validation dataset paths after `--train_data_dir` and `--valid_data_dir`
with the path of your downloaded data. `--fix_ori` normalizes the orientation of
all motion sequences in the dataset: for each motion sequence, rotate all poses in that sequence
so that the first pose faces towards some canonical orientation (e.g. pointing out of the screen)
and the motion sequence continues with the rotated first pose. `--experiment` specifies the name
of the current experiment. Running the above command, all model checkpoints and 
training logs will be saved at `<path_to_project_root>/training/default_exp`.

To test a model checkpoint, you can run
```
python test_contactformer.py ../data/proxd_valid/ --load_model ../training/contactformer/model_ckpt/best_model_recon_acc.pt --model_name contactformer --fix_ori --test_on_valid_set --output_dir PATH_OF_OUTPUT
```
The above command tests ContactFormer on the validation split of PROXD dataset.
The first argument is location of the validation set folder. 
`--model_name` is an arbitrary name you can set for disguishing the model you are testing. 
It can also help you pinpoint the location of the test result
since the result will be saved in a text file at the location `PATH_OF_OUTPUT/validation_results_<model_name>.txt`.

You can also run add a `--save_video` flag to save the visualization of contact label prediction
for some specific motion sequence. For example, you can run
```
python test_contactformer.py ../data/proxd_valid/ --load_model ../training/contactformer/model_ckpt/best_model_recon_acc.pt --model_name contactformer --fix_ori --single_seq_name MPH112_00151_01 --save_video --output_dir PATH_OF_OUTPUT
```
to save the visualization for predicted contact labels along with rendered body and scene meshes
for each frame in the MPH112_00151_01 motion sequence. The rendered frames will be saved at
`PATH_OF_OUTPUT/MPH112_00151_01/`. **Note that you need a screen to run this command.**

There are other parameters you can set to change the training scheme or the model architecture. Check
`train_contactformer.py` and `test_contactformer.py` for more details.


## Scene Synthesis
To fit best object based on **the most probossible** contact prediction, you can run the following,
**under the root directory of this repository**:
```
python fit_best_obj.py --sequence_name <sequence_name> --vertices_path <path_to_human_vertices> --contact_labels_path <path_to_contact_predictions> --output_dir <output_directory>
```
The fitting results for **all candidate objects** will be saved under 
`<output_directory>/<sequence_name>/fit_best_obj`. 
For each predicted object category, there is a json file called "best_obj_id.json" 
which stores the object ID that achieves the lowest optimization loss.

For example, suppose you have human vertices for motion sequence MPH11_00150_01 saved at
`data/proxd_valid/vertices/MPH11_00150_01_verts.npy` and contact predictions saved at
`predictions/proxd_valid/MPH11_00150_01.npy`, you can run
```
python fit_best_obj.py --sequence_name MPH11_00150_01 --vertices_path data/proxd_valid/vertices/MPH11_00150_01_verts.npy --contact_labels_path predictions/proxd_valid/MPH11_00150_01.npy --output_dir fitting_results
```
Fitting results will be saved under `fitting_results/MPH11_00150_01/fit_best_obj`.

There are several things noteworthy here:
- If it is the first time you run this optimization for some human motion sequence,
the script can take several minutes since it needs to estimate the SDF for human meshes in
all frames. The estimation will be saved to accelerate future inference.
- If you want to run this script on a server/cluster without a screen, please add
`os.environ['PYOPENGL_PLATFORM'] = 'egl'` in `fit_best_obj.py` after the line `import os`.
- If you want to use contact probability for each object category as input (i.e.
if you used `--save_probability` flag when generating the contact prediction), 
you can need to `--input_probability` flag at the end.

If you want to visualize the fitting result (i.e. recovered objects along with the human motion),
using the same example as mentioned above, you can run
```
python vis_fitting_results.py --fitting_results_path fitting_results/MPH11_00150_01 --vertices_path data/proxd_valid/vertices/MPH11_00150_01_verts.npy
```
The script will save rendered frames in `fitting_results/MPH11_00150_01/rendering`. 
**Note that you need a screen to run this command.**

Note that the candidate objects for fitting will be from the `3D_Future` directory in this repository, which is a subset of the [3D Future dataset](https://tianchi.aliyun.com/specials/promotion/alibaba-3d-future). You can modify the candidate objects by changing the contents of the `3D_Future` directory.

## Citation
If you find this work helpful, please consider citing:
```
@inproceedings{10.1145/3550469.3555426,
        author = {Ye, Sifan and Wang, Yixing and Li, Jiaman and Park, Dennis and Liu, C. Karen and Xu, Huazhe and Wu, Jiajun},
        title = {Scene Synthesis from Human Motion},
        year = {2022},
        isbn = {9781450394703},
        publisher = {Association for Computing Machinery},
        address = {New York, NY, USA},
        url = {https://doi.org/10.1145/3550469.3555426},
        doi = {10.1145/3550469.3555426},
        booktitle = {SIGGRAPH Asia 2022 Conference Papers},
        articleno = {26},
        numpages = {9},
        keywords = {Scene synthesis, activity understanding, motion analysis},
        location = {Daegu, Republic of Korea},
        series = {SA '22}
        }
}
```