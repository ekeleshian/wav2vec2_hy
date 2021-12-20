# Training an Armenian Wav2Vec2 Model (baseline)

*Important note: This generates just a baseline model, meaning it transforms the data into the appropriate format for wav2vec2 and sets the proper configs for GPU compatibility. Basically, baseline means the code works and moreover the code works on a high compute machine.  Optimizations around the data preparation and hyperparams for training are not included.*

### Prerequisites
* Download the armenian dataset from [commonvoice](https://commonvoice.mozilla.org/en/datasets)
* The version of torch in the requirements assumes your OS is Linux, your package management is Pip, and your compute platform includes CUDA 11.3.  I used GCP's Deep Learning VM Image, preconfigured with the following: 
	-  GPU Type: NVIDIA Tesla T4
	-  Machine Type: n1-standard-2 (2 vCPU, 7.5 GB memory)
	-  Framework: Tensorflow Enterprise 2.7 (CUDA 11.3)

### Getting started

1. Install all the requirements in a new virtual env.
```bash
cat requirements.txt | xargs -n 1 -L 1 pip install 
```

2. Download the [Armenian Common Voice dataset](https://commonvoice.mozilla.org/en/datasets) and move the unzipped folder to the root of repo. In addition, create the following dirs from the root of the project's repo:
```bash
mkdir cv-corpus-7.0-2021-07-21/hy-AM/wav_clips_32/
```

```bash
cv-corpus-7.0-2021-07-21/hy-AM/wav_clips_16/
```
 

3.  Prepare the dataset to make it compatible with the kind of input wav2vec2 expects.  This means to downsample the audio files to a sampling rate of 16000, tokenize the text, normalize the speech vectors, i.e. run the following command:
```bash
python prepare_dataset_hy.py
```

* A few files will be written to disk once this code is done running, notably the pickled files which will be needed for training: `prepared_test_hy.pkl`, `prepared_train_hy.pkl`, `processor_hy.pkl`.  

4.  Train the data, i.e. run the following command: 
```bash
python train.py
```

5.  [TBD] [##TODO] [##WIP] Evaluate the test set. 


Training Results for Baseline:
```
{'eval_loss': 1.0382163524627686, 'eval_wer': 0.7918918918918919, 'eval_runtime': 11.7239, 'eval_samples_per_second': 8.444, 'epoch': 42.0}
```





