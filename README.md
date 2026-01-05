# DSCformer: A Dual-Branch Network for Crack Segmentation

**DSCformer** is a hybrid model integrating **Enhanced Dynamic Snake Convolution (Enhanced DSConv)** and **SegFormer** for crack segmentation in concrete structures.

## Abstract

Accurately detecting and segmenting cracks in concrete structures is essential for construction quality monitoring, ensuring safety and timely maintenance. While Convolutional Neural Networks (CNNs) have demonstrated strong performance in crack segmentation tasks, they often struggle with complex backgrounds and fine-grained tubular structures. Transformers, though excellent at capturing global context, tend to lack precision in fine feature extraction. 

To address these challenges, we introduce **DSCformer**, a hybrid model that combines the power of an **Enhanced Dynamic Snake Convolution (Enhanced DSConv)** with a Transformer architecture. 

### Key Contributions:
- **Enhanced DSConv**: Utilizes a pyramid kernel and bidirectional offset iteration to improve the capture of intricate crack structures.
- **Weighted Channel Attention Module**: Refines channel attention, ensuring more precise feature extraction.
- **DSCformer**: Combines nhanced DSConv and SegFormer, significantly improving crack segmentation performance and outperforming state-of-the-art methods.

We evaluate DSCformer on the **Crack3238** and **FIND** datasets, achieving impressive Intersection over Union (IoU) scores of **59.22%** and **87.24%**, respectively.

## Dataset Links

- **Crack3238 Dataset**: [Download Link](https://drive.google.com/file/d/1uhliXpfhiI9BJ_2k4iINNJwUbdCNovSt/view?usp=sharing)
- **FIND Dataset**: [Download Link](https://zenodo.org/records/6383044)

### Dataset Conversion

Both datasets need to be converted into **h5py** format to work with DSCformer. You can use the provided `create_h5py.py` script for the conversion, or alternatively, modify the `dataset.py` file to fit your own dataset format.

## Model Download

To train **DSCformer**, you will need the **NVIDIA SegFormer B0** model. You can download the pre-trained model from:

- **SegFormer B0 Model**: [Download Link](https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512/tree/main)

## Requirements

To run this code, make sure to install all required libraries by running:


	pip install -r requirements.txt


## Training DSCformer

Follow the steps below to train DSCformer on your dataset:
1. Convert your dataset into h5py format using the create_h5py.py script.
2. Download the pre-trained SegFormer B0 model.
3. Run the training script to train DSCformer on your dataset.

## License

This project is licensed under the MIT License. See the LICENSE￼ file for more details.
