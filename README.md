<img src="./images/logo.png" width="350">

# Facial-Expression-Recognition-Zoo (FER-Zoo)

[![hugging](https://img.shields.io/badge/🤗%20Hugging%20Face-Datasets-yellow)](https://huggingface.co/datasets/kdhht2334/DiffusionEmotion)
[![license](https://img.shields.io/badge/License-CC0/MIT-blue)](#licensing) <a href="https://releases.ubuntu.com/18.04/"><img alt="Ubuntu" src="https://img.shields.io/badge/Ubuntu-18.04-green"></a>
<a href="https://www.python.org/downloads/release/python-370/"><img alt="PyThon" src="https://img.shields.io/badge/Python-v3.8-blue"></a>
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>

FER-Zoo is a PyTorch toolbox for facial expression recognition (FER). Especially, we focus on affect estimation methods to regress valence-arousal (VA) value. This repository contains state-of-the-art (SOTA) FER frameworks as follows:


| Methods | Venue | Link |
| --- | --- | --- |
| Baseline | TAC 2017 | [[link]](https://arxiv.org/abs/1708.03985) |
| CAF | AAAI 2022 | [[link]](https://ojs.aaai.org/index.php/AAAI/article/download/16743/16550) |
| AVCE | ECCV 2022 | [[link]](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136730181.pdf) |
| ELIM | NeurIPS 2022 | [[link]](https://arxiv.org/pdf/2209.12172) |


What's New
---
- [Mar. 2023] Add open DiffusionExpression dataset created by `StableDiffusion-WebUI` [[link]](https://github.com/camenduru/stable-diffusion-webui-colab).
- [Mar. 2023] Add `Baseline` framework.
- [Mar. 2023] Add training and evaluation part of FER frameworks.
- [Mar. 2023] Initial version of FER-Zoo.


Requirements
---
* python >= 3.8.0
* pytorch >= 1.7.1
* torchvision >= 0.8.0
* pretrainedmodels >=0.7.4
* fabulous >= 0.4.0
* wandb > 0.13.0


Get Started
---

DiffusionExpression dataset is available at [🤗 Hugging Face Datasets](https://huggingface.co/datasets/kdhht2334/DiffusionEmotion).

#### Subsets
(TBD)

#### Data structure
(TBD)


Custom Datasets
---

1. You can utilize the images containing various expression patterns in the `custom_dataset` folder. These images were created from `StableDiffusion-WebUI` [[link]](https://github.com/camenduru/stable-diffusion-webui-colab).
  - Facial regions are cropped by using `facenet-pytorch` [[link]](https://github.com/timesler/facenet-pytorch).
  
2. Change `<dataset_type>` to `custom` when training the model using `run.sh`.


Public Datasets
---

1. Download four public benchmarks for training and evaluation (please download after **agreement** accepted).

  - [AffectNet](http://mohammadmahoor.com/affectnet/)
  - [Aff-wild](https://ibug.doc.ic.ac.uk/resources/first-affect-wild-challenge/) 
  - [Aff-wild2](https://ibug.doc.ic.ac.uk/resources/aff-wild2/)
  - [AFEW-VA](https://ibug.doc.ic.ac.uk/resources/afew-va-database/)
 
 (For more details visit [website](https://ibug.doc.ic.ac.uk/))

2. Follow preprocessing rules for each dataset by referring pytorch official [custom dataset tutorial](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html).


Training
---

Just run the below script!
```
chmod 755 run.sh
./run.sh <dataset_type> <method> <gpu_no> <port_no> 
```
- `<dataset_type>`: 2 options (`custom` or `public`).
- `<method>`: 4 options (`elim`, `avce`, `caf`, and `baseline`).
- `<gpu_no>`: GPU number such as 0 (or 0, 1 etc.)
- `<port_no>`: port number to clarify workers (e.g., 12345)


Evaluation
---

- Evaluation is performed automatically at each `print_check` point in training phase.


Milestone
---
  - [x] Build SOTA FER frameworks
  - [ ] Upload pre-trained weights
  - [ ] Bench-marking table


Acknowledgments
This repository is partially inspired by [FaceX-Zoo](https://github.com/JDAI-CV/FaceX-Zoo) and [InsightFace_Pytorch](https://github.com/TreB1eN/InsightFace_Pytorch).

Citation
---

If our work is useful for your work, then please consider citing below bibtex:

```bibtex
@inproceedings{kim2021contrastive,
	title={Contrastive adversarial learning for person independent facial emotion recognition},
    author={Kim, Daeha and Song, Byung Cheol},
    booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
    volume={35},
    number={7},
    pages={5948--5956},
    year={2021}
}

@inproceedings{kim2022emotion,
    title={Emotion-aware Multi-view Contrastive Learning for Facial Emotion Recognition},
    author={Kim, Daeha and Song, Byung Cheol},
    booktitle={Computer Vision--ECCV 2022: 17th European Conference, Tel Aviv, Israel, October 23--27, 2022, Proceedings, Part XIII},
    pages={178--195},
    year={2022},
    organization={Springer}
}

@misc{kim2022elim,
        author = {Kim, Daeha and Song, Byung Cheol},
        title = {Optimal Transport-based Identity Matching for Identity-invariant Facial Expression Recognition},
        Year = {2022},
        Eprint = {arXiv:2209.12172}
}
```

Contact (or collaborate)
---

If you have any questions or work with me, feel free to contact [Daeha Kim](kdhht5022@gmail.com).


Licensing
---

The ExpressionDiffusion dataset is available under the [CC0 1.0 License](https://creativecommons.org/publicdomain/zero/1.0/).
The Python code in this repository is available under the [MIT License](./LICENSE).