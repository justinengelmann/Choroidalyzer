# Choroidalyzer: An open-source, end-to-end pipeline for choroidal analysis in optical coherence tomography

## Choroidalyzer
![image](https://github.com/justinengelmann/Choroidalyzer/assets/43992544/86399fc5-1acd-4128-ac4d-6dc106bbc064)

Choroidalyzer is an open-source, end-to-end pipeline for analysing the choroid in retinal optical coherence tomography (OCT). Concretely, it outputs choroid thickness, area, and vascular index. To do that, Choroidalyzer first segments the choroid region and vessels, and identifies the fovea location using a neural network, and then uses those the calculate the metrics. All of that happens fully-automatically.

## Setup

We recommend to first create a conda environment. If you haven't installed it on your computer already, we recommend the lightweight [miniconda](https://docs.conda.io/projects/miniconda/en/latest/). Then create an environment for Choroidalyzer and activate it:

```
conda create -n choroidalyzer python=3.10

conda activate choroidalyzer
```

Next, install the necessary dependencies.

```
pip install torch torchvision torchaudio matplotlib tqdm pandas scikit-image scipy
```

If you want to use jupyter lab, optionally also install that.

```
pip install jupyterlab
```

Download the code files from this repo, or "git clone" it.

Choroidalyzer will automagically download the model weights the first time you run it and then use them every time you use it in the future.

## Basic Usage

```python
from choroidalyze import Choroidalyzer

# initialise choroidalyzer
choroidalyzer = Choroidalyzer()

# replace with the path to your image
metrics = choroidalyzer.analyze('example_data/image1.png')
print(metrics)
# prints: {'thickness': 183.0, 'area': 1.356311, 'vascular_index': 0.46774, 'vessel_area': 0.634401, 'raw_thickness': array([174, 263, 112])}
```

*If you have any trouble running Choroidalyzer, please reach out and we'll be glad to help!*

You can also access the raw segmentations with choroidalyzer.predict() or visualise them with choroidalyzer.predict_and_plot(). See the [Example Usage](https://github.com/justinengelmann/Choroidalyzer/blob/main/ExampleUsage.ipynb) notebook for more examples.

## Advanced Usage

### Pixel scaling
The physical size of pixels in OCT B-Scans is not constant and instead can vary between scans. Ideally, you specify the value for each scan which you can extract from the metadata of the original OCT file. Otherwise, if no pixel-scaling is available, Choroidalyzer will fallback on a default value of (11.49, 3.87).

You can either set a new default to be used for all images:
```python
choroidalyzer = Choroidalyzer(default_scale=(value1, value2))
metrics = choroidalyzer.analyze('example_data/image1.png')
```
or for each image you analyse:
```python
choroidalyzer = Choroidalyzer()
metrics = choroidalyzer.analyze('example_data/image1.png', scale=(value1, value2))
```

### GPU inference
If you have a (recent-ish Nvidia) GPU, the neural network that powers the Choroidalyzer segmentations can be GPU-accelerated by setting the device to "cuda". In our experience, Choroidalyzer is very, very fast even on CPU, so this is mainly needed if you want to analyse large quantities of data.
```python
choroidalyzer = Choroidalyzer(device='cuda')
```

### Thresholds
The default binarisation thresholds are 0.5 for region and vessels, and 0.1 for the fovea which is only used for visualisation purposes. These work quite well for most images. However, if you have very difficult images, changing the thresholds could help a little bit.

You can set new default thresholds when initialising Choroidalyzer:
```python
choroidalyzer = Choroidalyzer(default_thresholds=(region_thresh, vessel_thresh, fovea_thresh))
```
or override the current default for a specific image:
```python
choroidalyzer = Choroidalyzer()
metrics = choroidalyzer.analyze('example_data/image1.png', thresholds=(region_thresh, vessel_thresh, fovea_thresh))
```




