# MTRSAP - Multimodal Transformer for Real-time Surgical Activity Recognition and Prediction

This repository provides the code developed for paper "Multimodal Transformer for Real-time Surgical Activity Recognition and Prediction" submitted to ICRA 2024.

## Table of Contents

- [Project Title](#project-title)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
  - [Usage](#usage)
  - [Contributing](#contributing)
  - [License](#license)
  - [Acknowledgements](#acknowledgements)

## Introduction

This repo is the official code for the ICRA 2024 paper "Multimodal Transformer for Real-time Surgical Activity Recognition and Prediction"

## Getting Started

Please follow the below instructions to setup the code in your environment.

### Prerequisites

1. **Anaconda**: Make sure to have Anaconda installed on your system. You can download it from Anaconda's official website.

2. **Preprocessed Dataset**: Obtain the preprocessed dataset required for your project. Refer to the Usage section for detailed instructions on acquiring and incorporating the dataset.

3. **Operating System**: While the project is designed to be compatible with various operating systems, Ubuntu is the preferred environment.


### Installation

1. Create the conda environment using the environment file. ``` conda env create -f environment.yml```
2. Verify PyTorch was installed correclty.
3. Place the preprocessed data in the **ProcessedData**.
4. Verify the configuration is as required in **config.py**. Learning parameters are defined in **config.py**.

## Usage

To reproduce gesture recognition results use the following command with the original configuration.

``` python train_recognition.py --model transformer --dataloader v2 --modality 16 ```

## Contributing

Explain how others can contribute to your project. Include guidelines for submitting bug reports or feature requests.

## License

Specify the license under which your project is distributed. For example, [MIT License](https://opensource.org/licenses/MIT).

## Acknowledgements

Give credit to other projects, tutorials, or resources that inspired or helped your project.

