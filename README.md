 # Egocentric Online Action Segmentation with Behavior-Centred Feature Augmentation

This is the repository for the paper "Egocentric Online Action Segmentation with Behavior-Centred Feature Augmentation".

The Egocentric Online Action Segmentation (EOAS) task aims to sequentially segment untrimmed egocentric videos into distinct action segments in a streaming manner. 
Previous methods primarily focused on improving contextual information utilization, which highly relied on leveraging the prior context. 
However, under online constraints, the absence of post context limits the effectiveness of the prior context in learning the action semantics. 
Excessive reliance on prior context may lead to insufficient feature representations of current presented behavior. 
<div align="center">
   <img src="readme_file/framework.png" alt="case 1" width="600">
   <br><br>
</div>
To tackle this problem, we propose a novel EOAS method, termed Behavior-Centred Feature Augmentation (BCFA) , which consists of two key modules:
(1) Behavior Prototype Learning utilizes prototypes to model the common sense of each action across different surroundings, enhancing the model's ability to capture the shared characteristics of behaviors. 
(2) Presented Behavior Enhancement leveraged both the intrinsic characteristics of the current presented behavior itself and the common sense captured by BPL for feature enhancement, mitigating the absence of post contextualization. 
We evaluate our proposed BCFA method on three public EOAS benchmark datasets, GTEA, EgoProceL, and EgoPER, and demonstrate that our proposed BCFA approach outperforms recent state-of-the-art methods.


## Visualization
<div align="center">
   <img src="readme_file/tea.png" alt="case 1" width="600">
   <br><br>
   <img src="readme_file/pinwheels.png" alt="case 2" width="600">
   <br><br>
</div>

For more experimental details and visualizations, please refer to our main text and supplementary materials.

## Repository Structure

- **main.py**: Script to train and evaluate the model.
- **BCFA.py**: Contains the implementation of the neural network models.
- **batch_gen.py**: Script for generating batches of data for training and evaluation.
- **eval.py**: Evaluation script.
- **data/**: Directory containing datasets, including ground truth and feature files.

## Data
**GTEA**: download GTEA data from [link1](https://zenodo.org/records/3625992#.Xiv9jGhKhPY) or [link2](https://mega.nz/#!O6wXlSTS!wcEoDT4Ctq5HRq_hV-aWeVF1_JB3cacQBQqOLjCIbc8). Please refer to [ms-tcn](https://github.com/yabufarha/ms-tcn) or [CVPR2024-FACT](https://github.com/ZijiaLewisLu/CVPR2024-FACT).  
**EgoProceL**: download EgoProceL data from [G-Drive](https://drive.google.com/drive/folders/1qYPLb7Flcl0kZWXFghdEpvrrkTF2SBrH). Please refer to [CVPR2024-FACT](https://github.com/ZijiaLewisLu/CVPR2024-FACT).  
**EgoPER**: download EgoPER data from [G-Drive](https://drive.google.com/drive/folders/1xZKJTme1FITMHKB3W_jMutFZV6O3pPDV?usp=sharing). Please refer to [EgoPER](https://www.khoury.northeastern.edu/home/eelhami/egoper.htm) for the original data. 

## Download the checkpoints
We have uploaded the model training checkpoints and the necessary data for testing.  
Please ensure the checkpoint is saved to the directory **models** and the data file is placed in the **data** folder.  

If you download them to other locations, update the corresponding arguments in `main.py` accordingly.






   you can get our checkpoint at [BCFA-checkpoint](https://drive.google.com/drive/folders/1u2_7aE-QfgWdztLoc7tZ9zxjmUFL0FwV?usp=drive_link)
   We will release other files after reviewing.
## Testing
To test the model, use the following command:

```
python main.py --action predict --dataset <dataset_name> --split <split_number> --exp_id BCFA --causal --graph  --balancing_lw <balancing_lw> --be_lw <be_lw> --short_window_scale <short_window_scale>
```

**Note:** Theoretically, to test the model in an online setting, you should use the `--action predict_online` argument, which makes predictions frame by frame. However, if the model is set to be causal, it will only make predictions based on frames up to the current frame. In this case, using `--action predict` will produce the same results while being more efficient.

## Experimental Results on EgoProceL, EgoPER, and GTEA Datasets

<h3 style="text-align:center;">Results on EgoProceL Dataset</h3>

<table style="width:100%; border-collapse:collapse; text-align:center; margin:auto; border:1px solid black;">
    <thead>
        <tr>
            <th style="width:30%; text-align:center; border:1px solid black;">Method</th>
            <th style="width:14%; text-align:center; border:1px solid black;">Acc</th>
            <th style="width:14%; text-align:center; border:1px solid black;">Edit</th>
            <th style="width:14%; text-align:center; border:1px solid black;">F1@0.1</th>
            <th style="width:14%; text-align:center; border:1px solid black;">F1@0.25</th>
            <th style="width:14%; text-align:center; border:1px solid black;">F1@0.5</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td style="border:1px solid black;">MSTCN (CVPR'19)</td>
            <td style="border:1px solid black;">64.5</td>
            <td style="border:1px solid black;">42.5</td>
            <td style="border:1px solid black;">45.2</td>
            <td style="border:1px solid black;">41.6</td>
            <td style="border:1px solid black;">33.0</td>
        </tr>
        <tr>
            <td style="border:1px solid black;">ASFormer (BMVC'21)</td>
            <td style="border:1px solid black;">64.8</td>
            <td style="border:1px solid black;">48.1</td>
            <td style="border:1px solid black;">49.8</td>
            <td style="border:1px solid black;">45.0</td>
            <td style="border:1px solid black;">35.4</td>
        </tr>
        <tr>
            <td style="border:1px solid black;">OODL (CVPR'22)</td>
            <td style="border:1px solid black;">66.4</td>
            <td style="border:1px solid black;">44.0</td>
            <td style="border:1px solid black;">44.7</td>
            <td style="border:1px solid black;">41.5</td>
            <td style="border:1px solid black;">30.5</td>
        </tr>
        <tr>
            <td style="border:1px solid black;">ProTAS (CVPR'24)</td>
            <td style="border:1px solid black;">68.5</td>
            <td style="border:1px solid black;">52.1</td>
            <td style="border:1px solid black;">51.6</td>
            <td style="border:1px solid black;">48.6</td>
            <td style="border:1px solid black;">36.8</td>
        </tr>
        <tr>
            <td style="border:1px solid black;"><b>BCFA (Ours)</b></td>
            <td style="border:1px solid black;"><b>69.1</b></td>
            <td style="border:1px solid black;"><b>56.9</b></td>
            <td style="border:1px solid black;"><b>56.3</b></td>
            <td style="border:1px solid black;"><b>53.0</b></td>
            <td style="border:1px solid black;"><b>41.0</b></td>
        </tr>
    </tbody>
</table>

<h3 style="text-align:center;">Results on EgoPER Dataset</h3>

<table style="width:100%; border-collapse:collapse; text-align:center; margin:auto; border:1px solid black;">
    <thead>
        <tr>
            <th style="width:30%; text-align:center; border:1px solid black;">Method</th>
            <th style="width:14%; text-align:center; border:1px solid black;">Acc</th>
            <th style="width:14%; text-align:center; border:1px solid black;">Edit</th>
            <th style="width:14%; text-align:center; border:1px solid black;">F1@0.1</th>
            <th style="width:14%; text-align:center; border:1px solid black;">F1@0.25</th>
            <th style="width:14%; text-align:center; border:1px solid black;">F1@0.5</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td style="border:1px solid black;">MSTCN (CVPR'19)</td>
            <td style="border:1px solid black;">71.8</td>
            <td style="border:1px solid black;">48.9</td>
            <td style="border:1px solid black;">56.2</td>
            <td style="border:1px solid black;">52.2</td>
            <td style="border:1px solid black;">39.4</td>
        </tr>
        <tr>
            <td style="border:1px solid black;">ASFormer (BMVC'21)</td>
            <td style="border:1px solid black;">70.3</td>
            <td style="border:1px solid black;">60.6</td>
            <td style="border:1px solid black;">66.1</td>
            <td style="border:1px solid black;">62.3</td>
            <td style="border:1px solid black;">44.7</td>
        </tr>
        <tr>
            <td style="border:1px solid black;">OODL (CVPR'22)</td>
            <td style="border:1px solid black;">71.2</td>
            <td style="border:1px solid black;">49.3</td>
            <td style="border:1px solid black;">55.6</td>
            <td style="border:1px solid black;">52.3</td>
            <td style="border:1px solid black;">40.0</td>
        </tr>
        <tr>
            <td style="border:1px solid black;">ProTAS (CVPR'24)</td>
            <td style="border:1px solid black;">71.7</td>
            <td style="border:1px solid black;">62.4</td>
            <td style="border:1px solid black;">68.8</td>
            <td style="border:1px solid black;">65.9</td>
            <td style="border:1px solid black;">48.6</td>
        </tr>
        <tr>
            <td style="border:1px solid black;"><b>BCFA (Ours)</b></td>
            <td style="border:1px solid black;"><b>76.2</b></td>
            <td style="border:1px solid black;"><b>72.3</b></td>
            <td style="border:1px solid black;"><b>73.3</b></td>
            <td style="border:1px solid black;"><b>70.4</b></td>
            <td style="border:1px solid black;"><b>58.4</b></td>
        </tr>
    </tbody>
</table>

<h3 style="text-align:center;">Results on GTEA Dataset</h3>

<table style="width:100%; border-collapse:collapse; text-align:center; margin:auto; border:1px solid black;">
    <thead>
        <tr>
            <th style="width:30%; text-align:center; border:1px solid black;">Method</th>
            <th style="width:14%; text-align:center; border:1px solid black;">Acc</th>
            <th style="width:14%; text-align:center; border:1px solid black;">Edit</th>
            <th style="width:14%; text-align:center; border:1px solid black;">F1@0.1</th>
            <th style="width:14%; text-align:center; border:1px solid black;">F1@0.25</th>
            <th style="width:14%; text-align:center; border:1px solid black;">F1@0.5</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td style="border:1px solid black;">MSTCN (CVPR'19)</td>
            <td style="border:1px solid black;">74.0</td>
            <td style="border:1px solid black;">64.4</td>
            <td style="border:1px solid black;">71.8</td>
            <td style="border:1px solid black;">69.4</td>
            <td style="border:1px solid black;">56.0</td>
        </tr>
        <tr>
            <td style="border:1px solid black;">ASFormer (BMVC'21)</td>
            <td style="border:1px solid black;">77.2</td>
            <td style="border:1px solid black;">73.3</td>
            <td style="border:1px solid black;">79.6</td>
            <td style="border:1px solid black;">77.1</td>
            <td style="border:1px solid black;">65.0</td>
        </tr>
        <tr>
            <td style="border:1px solid black;">OODL (CVPR'22)</td>
            <td style="border:1px solid black;">74.0</td>
            <td style="border:1px solid black;">64.7</td>
            <td style="border:1px solid black;">70.3</td>
            <td style="border:1px solid black;">66.9</td>
            <td style="border:1px solid black;">54.1</td>
        </tr>
        <tr>
            <td style="border:1px solid black;">ProTAS (CVPR'24)</td>
            <td style="border:1px solid black;">77.0</td>
            <td style="border:1px solid black;">74.1</td>
            <td style="border:1px solid black;">80.2</td>
            <td style="border:1px solid black;">77.5</td>
            <td style="border:1px solid black;">66.1</td>
        </tr>
        <tr>
            <td style="border:1px solid black;"><b>BCFA (Ours)</b></td>
            <td style="border:1px solid black;"><b>77.3</b></td>
            <td style="border:1px solid black;"><b>78.6</b></td>
            <td style="border:1px solid black;"><b>82.6</b></td>
            <td style="border:1px solid black;"><b>79.4</b></td>
            <td style="border:1px solid black;"><b>67.6</b></td>
        </tr>
    </tbody>
</table>


