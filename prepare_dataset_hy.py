import json
import pickle
import random
import re
from datasets import Dataset
import numpy as np
import pandas as pd
import soundfile as sf
from transformers import Wav2Vec2CTCTokenizer
from transformers import Wav2Vec2FeatureExtractor
from transformers import Wav2Vec2Processor
import mp3_to_wav_downsample


CHARS_TO_IGNORE_REGEX = '[՝։՜…–—.՚(),՞«»]'


def to_dataframe(tsv_path):
	df = pd.read_csv(tsv_path, sep='\t')
	df.drop(columns=['up_votes','age', 'accent', 'locale', 'segment', 'gender', 'client_id'], inplace=True)
	df.drop(index=df[df['down_votes'] > 0].index, inplace=True)
	df.reset_index(inplace=True)
	df.drop(columns=['down_votes'], inplace=True)
	return df


def preprocess_text(df):
	df['sentence'] = df['sentence'].apply(lambda x: re.sub(CHARS_TO_IGNORE_REGEX, "", x.lower()))
	return df


def generate_vocab(df):
	def extract_all_chars(batch):
		all_text = " ".join(batch['sentence'])
		vocab = list(set(all_text))
		return {'vocab': [vocab], 'all_text': [all_text]}

	vocabs = extract_all_chars(df)
	vocab_list = list(set(vocabs['vocab'][0]))
	vocab_dict = { v: k for k, v in enumerate(vocab_list)}
	vocab_dict["|"] = vocab_dict[" "]
	del vocab_dict[" "]
	vocab_dict['[UNK]'] = len(vocab_dict)
	vocab_dict['[PAD]'] = len(vocab_dict)

	vocab_filename = 'vocab_hy.json'

	with open(vocab_filename, 'w', encoding='utf8') as vocab_file:
		json.dump(vocab_dict, vocab_file, ensure_ascii=False)

	print(f'{vocab_filename} is generated and stored in root of project folder')


def change_audio_path(df):
	def new_path(path):
		return f"/Users/elizabethkeleshian/wav2vec2-huggingface/cv-corpus-7.0-2021-07-21/hy-AM/wav_clips_16/{path.replace('.mp3', '.wav')}"
	
	df['new_path'] = df['path'].apply(new_path)
	return df


def speech_file_to_array_fn(batch):
    speech_array, sampling_rate = sf.read(batch['new_path'])
    batch['speech'] = speech_array
    batch['sampling_rate'] = sampling_rate
    batch['target_text'] = batch['sentence']
    return batch


def perform_sanity_check(df):
	rand_int = random.randint(0, len(df))
	target_text = df.iloc[rand_int]['target_text']
	speech_vector = np.asarray(df.iloc[rand_int]['speech'])
	sampling_rate = df.iloc[rand_int]['sampling_rate']
	print("target text: ", target_text)
	assert isinstance(target_text, str), "target_text should be string"
	no_ignored_chars = True
	for char in CHARS_TO_IGNORE_REGEX:
		if char in target_text:
			no_ignored_chars = False
			break
	assert no_ignored_chars, "target_text should not have any of the ignored chars"
	assert target_text.islower(), "target_text should be in lower case"
	print('input array shape: ', speech_vector.shape)
	assert len(speech_vector.shape) == 1, "speech array should be a 1D vector"
	print('sampling rate: ', sampling_rate, '\n')
	assert sampling_rate == 16000, "sampling rate should be 16000"



def prepare_inputs_for_training(df):
	tokenizer = Wav2Vec2CTCTokenizer('./vocab_hy.json', unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
	feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, return_attention_mask=False)
	processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)


	def prepare_dataset(batch):
		assert(
			len(set(batch['sampling_rate'])) == 1
			), f"Make sure all inputs have the same sampling rate of {processor.feature_extractor.sampling_rate}"
		input_values = []
		for row in batch.iterrows():
			input_values.append(processor(row[1].speech, sampling_rate=16000).input_values)
		batch['input_values'] = input_values
		input_ids = []
		with processor.as_target_processor():
			for row in batch.iterrows():
				input_ids.append(processor(row[1]['target_text']).input_ids)
			batch['labels'] = input_ids

		return batch


	def split_data_from_dict():
		train_len = int(0.7*len(df))
		df_idx = list(range(len(df)))
		train_idx = sorted(random.sample(df_idx, train_len))
		train_df = df.iloc[train_idx]
		test_df = df.loc[~df.index.isin(train_idx)]
		train_dict = dict()
		test_dict = dict()
		train_dict['input_values'] = train_df['input_values']
		train_dict['labels'] = train_df['labels']
		test_dict['input_values'] = test_df['input_values']
		test_dict['labels'] = test_df['labels']
		train_dict = Dataset.from_dict(train_dict)
		test_dict = Dataset.from_dict(test_dict)
		return train_dict, test_dict


	df = prepare_dataset(df)
	df['input_values'] = df['input_values'].apply(lambda x: x[0])
	train_dict, test_dict = split_data_from_dict()

	with open('processor_hy.pkl', 'wb') as fi:
		pickle.dump(processor, fi)

	with open('prepared_train_hy.pkl', 'wb') as fi:
		pickle.dump(train_dict, fi)

	with open('prepared_test_hy.pkl', 'wb') as fi:
		pickle.dump(test_dict, fi)
	
	return train_dict, test_dict


if __name__ == "__main__":
	df = to_dataframe('cv-corpus-7.0-2021-07-21/hy-AM/train.tsv')
	df = preprocess_text(df)
	generate_vocab(df)
	# mp3_to_wav_downsample.prepare_downsample()
	df = change_audio_path(df)
	df = df.apply(speech_file_to_array_fn, axis=1)
	perform_sanity_check(df)
	train_dict, test_dict = prepare_inputs_for_training(df)




