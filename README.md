 # Evidential Temporal Contextualization for Egocentric Online Action Segmentation

This is the repository for the paper "Egocentric Online Action Segmentation with Behavior-Centred Feature Augmentation".

The Egocentric Online Action Segmentation (EOAS) task aims to sequentially segment untrimmed egocentric videos into distinct action segments in a streaming manner. 
Previous methods primarily focused on improving contextual information utilization, which highly relied on leveraging the prior context. 
However, under online constraints, the absence of post context limits the effectiveness of the prior context in learning the action semantics. 
Excessive reliance on prior context may lead to insufficient feature representations of current presented behavior. 
To tackle this problem, we propose a novel EOAS method, termed \textit{\textbf{B}ehavior-\textbf{C}entred \textbf{F}eature \textbf{A}ugmentation (BCFA)} , which consists of two key modules:
(1) Behavior Prototype Learning utilizes prototypes to model the common sense of each action across different surroundings, enhancing the model's ability to capture the shared characteristics of behaviors. 
(2) Presented Behavior Enhancement leveraged both the intrinsic characteristics of the current presented behavior itself and the common sense captured by BPL for feature enhancement, mitigating the absence of post contextualization. 
We evaluate our proposed BCFA method on three public EOAS benchmark datasets, GTEA, EgoProceL, and EgoPER, and demonstrate that our proposed BCFA approach outperforms recent state-of-the-art methods.

## Visualization

For more experimental details and visualizations, please refer to our main text and supplementary materials.

## Repository Structure

- **main.py**: Script to train and evaluate the model.
- **edl_loss**: Script to calculate the EDL loss.
- **BCFA.py**: Contains the implementation of the neural network models.
- **batch_gen.py**: Script for generating batches of data for training and evaluation.
- **eval.py**: Evaluation script.
- **data/**: Directory containing datasets, including ground truth and feature files.

## Data
**GTEA**: download GTEA data from [link1](https://zenodo.org/records/3625992#.Xiv9jGhKhPY) or [link2](https://mega.nz/#!O6wXlSTS!wcEoDT4Ctq5HRq_hV-aWeVF1_JB3cacQBQqOLjCIbc8). Please refer to [ms-tcn](https://github.com/yabufarha/ms-tcn) or [CVPR2024-FACT](https://github.com/ZijiaLewisLu/CVPR2024-FACT).  
**EgoProceL**: download EgoProceL data from [G-Drive](https://drive.google.com/drive/folders/1qYPLb7Flcl0kZWXFghdEpvrrkTF2SBrH). Please refer to [CVPR2024-FACT](https://github.com/ZijiaLewisLu/CVPR2024-FACT).  
**EgoPER**: download EgoPER data from [G-Drive](https://drive.google.com/drive/folders/1xZKJTme1FITMHKB3W_jMutFZV6O3pPDV?usp=sharing). Please refer to [EgoPER](https://www.khoury.northeastern.edu/home/eelhami/egoper.htm) for the original data. 

## Download the checkpoints
We have uploaded the checkpoints of model training and the data you need to test it.
   Please create and save the checkpoint to **models/<exp_id>/<dataset_name>/<split_number>** folder and the data file to **data** folder.
   
   If you download to other paths. Change the relative args in main.py.

   you can get our checkpoint at [G-Drive]
   We will release other files after reviewing.
## Testing
To test the model, use the following command:

```
python main.py --action predict --dataset <dataset_name> --split <split_number> --exp_id BCFA --causal --graph
```

**Note:** Theoretically, to test the model in an online setting, you should use the `--action predict_online` argument, which makes predictions frame by frame. However, if the model is set to be causal, it will only make predictions based on frames up to the current frame. In this case, using `--action predict` will produce the same results while being more efficient.



