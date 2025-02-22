# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import io
import os
from typing import Dict, List, Optional, Union

import braceexpand
import torch
import webdataset as wd

from nemo.collections.asr.parts.preprocessing.segment import available_formats as valid_sf_formats
from nemo.collections.common.parts.preprocessing import collections
from nemo.core.classes import Dataset, IterableDataset
from nemo.core.neural_types import AudioSignal, LabelsType, LengthsType, NeuralType, RegressionValuesType
from nemo.utils import logging

# List of valid file formats (prioritized by order of importance)
VALID_FILE_FORMATS = ';'.join(['wav', 'mp3', 'flac'] + [fmt.lower() for fmt in valid_sf_formats.keys()])


def repeat_signal(signal, sig_len, required_length):
    """repeat signal to make short signal to have required_length
    Args:
        signal (FloatTensor): input signal
        sig_len (LongTensor): length of input signal
        required_length(float) : length of generated signal
    Returns:
        signal (FloatTensor): generated signal of required_length by repeating itself.
    """
    repeat = int(required_length // sig_len)
    rem = int(required_length % sig_len)
    sub = signal[-rem:] if rem > 0 else torch.tensor([])
    rep_sig = torch.cat(repeat * [signal])
    signal = torch.cat((rep_sig, sub))
    return signal


def normalize(signal):
    """normalize signal
    Args:
        signal(FloatTensor): signal to be normalized.
    """
    signal_minusmean = signal - signal.mean()
    return signal_minusmean / signal_minusmean.abs().max()


def count_occurence(manifest_file_id):
    """Count number of wav files in Dict manifest_file_id. Use for _TarredAudioToLabelDataset.
    Args:
        manifest_file_id (Dict): Dict of files and their corresponding id. {'A-sub0' : 1, ..., 'S-sub10':100}
    Returns:
        count (Dict): Dict of wav files {'A' : 2, ..., 'S':10}
    """
    count = dict()
    for i in manifest_file_id:
        audio_filename = i.split("-sub")[0]
        count[audio_filename] = count.get(audio_filename, 0) + 1
    return count


def _speech_collate_fn(batch, pad_id):
    """collate batch of audio sig, audio len, tokens, tokens len
    Args:
        batch (Optional[FloatTensor], Optional[LongTensor], LongTensor,
               LongTensor):  A tuple of tuples of signal, signal lengths,
               encoded tokens, and encoded tokens length.  This collate func
               assumes the signals are 1d torch tensors (i.e. mono audio).
    """
    _, audio_lengths, _, tokens_lengths = zip(*batch)
    max_audio_len = 0
    has_audio = audio_lengths[0] is not None
    if has_audio:
        max_audio_len = max(audio_lengths).item()
    max_tokens_len = max(tokens_lengths).item()

    audio_signal, tokens = [], []
    for sig, sig_len, tokens_i, tokens_i_len in batch:
        if has_audio:
            sig_len = sig_len.item()
            if sig_len < max_audio_len:
                pad = (0, max_audio_len - sig_len)
                sig = torch.nn.functional.pad(sig, pad)
            audio_signal.append(sig)
        tokens_i_len = tokens_i_len.item()
        if tokens_i_len < max_tokens_len:
            pad = (0, max_tokens_len - tokens_i_len)
            tokens_i = torch.nn.functional.pad(tokens_i, pad, value=pad_id)
        tokens.append(tokens_i)

    if has_audio:
        audio_signal = torch.stack(audio_signal)
        audio_lengths = torch.stack(audio_lengths)
    else:
        audio_signal, audio_lengths = None, None
    tokens = torch.stack(tokens)
    tokens_lengths = torch.stack(tokens_lengths)

    return audio_signal, audio_lengths, tokens, tokens_lengths


def _fixed_seq_collate_fn(self, batch):
    """collate batch of audio sig, audio len, tokens, tokens len
        Args:
            batch (Optional[FloatTensor], Optional[LongTensor], LongTensor,
                LongTensor):  A tuple of tuples of signal, signal lengths,
                encoded tokens, and encoded tokens length.  This collate func
                assumes the signals are 1d torch tensors (i.e. mono audio).
        """
    fixed_length = self.featurizer.sample_rate * self.time_length
    _, audio_lengths, _, tokens_lengths = zip(*batch)

    has_audio = audio_lengths[0] is not None
    fixed_length = int(min(fixed_length, max(audio_lengths)))

    audio_signal, tokens, new_audio_lengths = [], [], []
    for sig, sig_len, tokens_i, _ in batch:
        if has_audio:
            sig_len = sig_len.item()
            chunck_len = sig_len - fixed_length

            if chunck_len < 0:
                repeat = fixed_length // sig_len
                rem = fixed_length % sig_len
                sub = sig[-rem:] if rem > 0 else torch.tensor([])
                rep_sig = torch.cat(repeat * [sig])
                signal = torch.cat((rep_sig, sub))
                new_audio_lengths.append(torch.tensor(fixed_length))
            else:
                start_idx = torch.randint(0, chunck_len, (1,)) if chunck_len else torch.tensor(0)
                end_idx = start_idx + fixed_length
                signal = sig[start_idx:end_idx]
                new_audio_lengths.append(torch.tensor(fixed_length))

            audio_signal.append(signal)
        tokens.append(tokens_i)

    if has_audio:
        audio_signal = torch.stack(audio_signal)
        audio_lengths = torch.stack(new_audio_lengths)
    else:
        audio_signal, audio_lengths = None, None
    tokens = torch.stack(tokens)
    tokens_lengths = torch.stack(tokens_lengths)

    return audio_signal, audio_lengths, tokens, tokens_lengths


def _sliced_seq_collate_fn(self, batch, masked=False):
    """collate batch of audio sig, audio len, tokens, tokens len
    Args:
        batch (Optional[FloatTensor], Optional[LongTensor], LongTensor,
            LongTensor):  A tuple of tuples of signal, signal lengths,
            encoded tokens, and encoded tokens length.  This collate func
            assumes the signals are 1d torch tensors (i.e. mono audio).
        masked: Apply true lengths as input to speaker extractor for masking effect
    """
    slice_length = self.featurizer.sample_rate * self.time_length
    _, audio_lengths, _, tokens_lengths = zip(*batch)
    has_audio = audio_lengths[0] is not None

    audio_signal, num_slices, tokens, audio_lengths = [], [], [], []
    for sig, sig_len, tokens_i, _ in batch:
        if has_audio:
            sig_len = sig_len.item()
            if len(sig) < slice_length:
                sig = repeat_signal(sig, sig_len, slice_length)
            audio_signal.append(sig)
            num_slices.append(1)
            tokens.append(1)
        if masked:
            audio_lengths.append(int(sig_len))
        else:
            audio_lengths.append(int(slice_length))

    if has_audio:
        audio_signal = torch.stack(audio_signal)
        audio_lengths = torch.tensor(audio_lengths)
    else:
        audio_signal, audio_lengths = None, None
    tokens = torch.tensor(tokens)
    tokens_lengths = torch.tensor(num_slices)  # each embedding length

    return audio_signal, audio_lengths, tokens, tokens_lengths


def _vad_frame_seq_collate_fn(self, batch):
    """collate batch of audio sig, audio len, tokens, tokens len
    Args:
        batch (Optional[FloatTensor], Optional[LongTensor], LongTensor,
            LongTensor):  A tuple of tuples of signal, signal lengths,
            encoded tokens, and encoded tokens length.  This collate func
            assumes the signals are 1d torch tensors (i.e. mono audio).
            batch size equals to 1.
    """
    slice_length = int(self.featurizer.sample_rate * self.time_length)
    _, audio_lengths, _, tokens_lengths = zip(*batch)
    slice_length = min(slice_length, max(audio_lengths))
    shift = int(self.featurizer.sample_rate * self.shift_length)
    has_audio = audio_lengths[0] is not None

    audio_signal, num_slices, tokens, audio_lengths = [], [], [], []

    append_len_start = slice_length // 2
    append_len_end = slice_length - slice_length // 2
    for sig, sig_len, tokens_i, _ in batch:
        if self.normalize_audio:
            sig = normalize(sig)
        start = torch.zeros(append_len_start)
        end = torch.zeros(append_len_end)
        sig = torch.cat((start, sig, end))
        sig_len += slice_length

        if has_audio:
            slices = torch.div(sig_len - slice_length, shift, rounding_mode='trunc')
            for slice_id in range(slices):
                start_idx = slice_id * shift
                end_idx = start_idx + slice_length
                signal = sig[start_idx:end_idx]
                audio_signal.append(signal)

            num_slices.append(slices)
            tokens.extend([tokens_i] * slices)
            audio_lengths.extend([slice_length] * slices)

    if has_audio:
        audio_signal = torch.stack(audio_signal)
        audio_lengths = torch.tensor(audio_lengths)
    else:
        audio_signal, audio_lengths = None, None

    tokens = torch.stack(tokens)
    tokens_lengths = torch.tensor(num_slices)
    return audio_signal, audio_lengths, tokens, tokens_lengths


class _AudioLabelDataset(Dataset):
    """
    Dataset that loads tensors via a json file containing paths to audio files,
    labels, and durations and offsets(in seconds). Each new line is a
    different sample. Example below:
    and their target labels. JSON files should be of the following format::
        {"audio_filepath": "/path/to/audio_wav_0.wav", "duration": time_in_sec_0, "label": \
target_label_0, "offset": offset_in_sec_0}
        ...
        {"audio_filepath": "/path/to/audio_wav_n.wav", "duration": time_in_sec_n, "label": \
target_label_n, "offset": offset_in_sec_n}
    Args:
        manifest_filepath (str): Dataset parameter. Path to JSON containing data.
        labels (list): Dataset parameter. List of target classes that can be output by the speaker recognition model.
        featurizer
        min_duration (float): Dataset parameter. All training files which have a duration less than min_duration
            are dropped. Note: Duration is read from the manifest JSON.
            Defaults to 0.1.
        max_duration (float): Dataset parameter.
            All training files which have a duration more than max_duration
            are dropped. Note: Duration is read from the manifest JSON.
            Defaults to None.
        trim (bool): Whether to use trim silence from beginning and end of audio signal using librosa.effects.trim().
            Defaults to False.
    """

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        """Returns definitions of module output ports.
        """

        output_types = {
            'audio_signal': NeuralType(
                ('B', 'T'),
                AudioSignal(freq=self._sample_rate)
                if self is not None and hasattr(self, '_sample_rate')
                else AudioSignal(),
            ),
            'a_sig_length': NeuralType(tuple('B'), LengthsType()),
        }

        if self.is_regression_task:
            output_types.update(
                {
                    'targets': NeuralType(tuple('B'), RegressionValuesType()),
                    'targets_length': NeuralType(tuple('B'), LengthsType()),
                }
            )
        else:

            output_types.update(
                {'label': NeuralType(tuple('B'), LabelsType()), 'label_length': NeuralType(tuple('B'), LengthsType()),}
            )

        return output_types

    def __init__(
        self,
        *,
        manifest_filepath: str,
        labels: List[str],
        featurizer,
        min_duration: Optional[float] = 0.1,
        max_duration: Optional[float] = None,
        trim: bool = False,
        is_regression_task: bool = False,
    ):
        super().__init__()
        self.collection = collections.ASRSpeechLabel(
            manifests_files=manifest_filepath.split(','),
            min_duration=min_duration,
            max_duration=max_duration,
            is_regression_task=is_regression_task,
        )

        self.featurizer = featurizer
        self.trim = trim
        self.is_regression_task = is_regression_task

        if not is_regression_task:
            self.labels = labels if labels else self.collection.uniq_labels
            self.num_classes = len(self.labels) if self.labels is not None else 1
            self.label2id, self.id2label = {}, {}
            for label_id, label in enumerate(self.labels):
                self.label2id[label] = label_id
                self.id2label[label_id] = label

            for idx in range(len(self.labels[:5])):
                logging.debug(" label id {} and its mapped label {}".format(idx, self.id2label[idx]))

        else:
            self.labels = []
            self.num_classes = 1

    def __len__(self):
        return len(self.collection)

    def __getitem__(self, index):
        sample = self.collection[index]

        offset = sample.offset

        if offset is None:
            offset = 0

        features = self.featurizer.process(sample.audio_file, offset=offset, duration=sample.duration, trim=self.trim)
        f, fl = features, torch.tensor(features.shape[0]).long()

        if not self.is_regression_task:
            t = torch.tensor(self.label2id[sample.label]).long()
        else:
            t = torch.tensor(sample.label).float()

        tl = torch.tensor(1).long()  # For compatibility with collate_fn used later

        return f, fl, t, tl


# Ported from https://github.com/NVIDIA/OpenSeq2Seq/blob/master/open_seq2seq/data/speech2text/speech_commands.py
class AudioToClassificationLabelDataset(_AudioLabelDataset):
    """
    Dataset that loads tensors via a json file containing paths to audio
    files, command class, and durations (in seconds). Each new line is a
    different sample. Example below:
    {"audio_filepath": "/path/to/audio_wav_0.wav", "duration": time_in_sec_0, "label": \
        target_label_0, "offset": offset_in_sec_0}
    ...
    {"audio_filepath": "/path/to/audio_wav_n.wav", "duration": time_in_sec_n, "label": \
        target_label_n, "offset": offset_in_sec_n}
    Args:
        manifest_filepath: Path to manifest json as described above. Can
            be comma-separated paths.
        labels (Optional[list]): String containing all the possible labels to map to
            if None then automatically picks from ASRSpeechLabel collection.
        featurizer: Initialized featurizer class that converts paths of
            audio to feature tensors
        max_duration: If audio exceeds this length, do not include in dataset
        min_duration: If audio is less than this length, do not include
            in dataset
        trim: Boolean flag whether to trim the audio
    """

    def _collate_fn(self, batch):
        return _speech_collate_fn(batch, pad_id=0)


class AudioToSpeechLabelDataset(_AudioLabelDataset):
    """
    Dataset that loads tensors via a json file containing paths to audio
    files, command class, and durations (in seconds). Each new line is a
    different sample. Example below:
    {"audio_filepath": "/path/to/audio_wav_0.wav", "duration": time_in_sec_0, "label": \
        target_label_0, "offset": offset_in_sec_0}
    ...
    {"audio_filepath": "/path/to/audio_wav_n.wav", "duration": time_in_sec_n, "label": \
        target_label_n, "offset": offset_in_sec_n}
    Args:
        manifest_filepath (str): Path to manifest json as described above. Can
            be comma-separated paths.
        labels (Optional[list]): String containing all the possible labels to map to
            if None then automatically picks from ASRSpeechLabel collection.
        min_duration (float): Dataset parameter.
            All training files which have a duration less than min_duration
            are dropped. Note: Duration is read from the manifest JSON.
            Defaults to 0.1.
        max_duration (float): Dataset parameter.
            All training files which have a duration more than max_duration
            are dropped. Note: Duration is read from the manifest JSON.
            Defaults to None.
        trim (bool): Whether to use trim silence from beginning and end
            of audio signal using librosa.effects.trim().
            Defaults to False.
        time_length (float): time length of slice (in seconds)
            Use this for speaker recognition and VAD tasks.
        shift_length (float): amount of shift of window for generating the frame for VAD task in a batch
            Use this for VAD task during inference.
        normalize_audio (bool): Whether to normalize audio signal.
            Defaults to False.
        is_regression_task (bool): Whether the dataset is for a regression task instead of classification
    """

    def __init__(
        self,
        *,
        manifest_filepath: str,
        labels: List[str],
        featurizer,
        min_duration: Optional[float] = 0.1,
        max_duration: Optional[float] = None,
        trim: bool = False,
        time_length: Optional[float] = 8,
        shift_length: Optional[float] = 1,
        normalize_audio: bool = False,
        is_regression_task: bool = False,
    ):
        logging.info("Time length considered for collate func is {}".format(time_length))
        logging.info("Shift length considered for collate func is {}".format(shift_length))
        self.time_length = time_length
        self.shift_length = shift_length
        self.normalize_audio = normalize_audio

        super().__init__(
            manifest_filepath=manifest_filepath,
            labels=labels,
            featurizer=featurizer,
            min_duration=min_duration,
            max_duration=max_duration,
            trim=trim,
            is_regression_task=is_regression_task,
        )

    def fixed_seq_collate_fn(self, batch):
        return _fixed_seq_collate_fn(self, batch)

    def sliced_seq_collate_fn(self, batch):
        return _sliced_seq_collate_fn(self, batch)

    def vad_frame_seq_collate_fn(self, batch):
        return _vad_frame_seq_collate_fn(self, batch)


class _TarredAudioLabelDataset(IterableDataset):
    """
    A similar Dataset to the AudioLabelDataSet, but which loads tarred audio files.

    Accepts a single comma-separated JSON manifest file (in the same style as for the AudioToSpeechLabelDataset),
    as well as the path(s) to the tarball(s) containing the wav files. Each line of the manifest should
    contain the information for one audio file, including at least the label and name of the audio
    file within the tarball.

    Valid formats for the audio_tar_filepaths argument include:
    (1) a single string that can be brace-expanded, e.g. 'path/to/audio.tar' or 'path/to/audio_{1..100}.tar.gz', or
    (2) a list of file paths that will not be brace-expanded, e.g. ['audio_1.tar', 'audio_2.tar', ...].

    Note: For brace expansion in (1), there may be cases where `{x..y}` syntax cannot be used due to shell interference.
    This occurs most commonly inside SLURM scripts. Therefore we provide a few equivalent replacements.
    Supported opening braces - { <=> (, [, < and the special tag _OP_.
    Supported closing braces - } <=> ), ], > and the special tag _CL_.
    For SLURM based tasks, we suggest the use of the special tags for ease of use.

    See the documentation for more information about accepted data and input formats.

    If using multiple processes the number of shards should be divisible by the number of workers to ensure an
    even split among workers. If it is not divisible, logging will give a warning but training will proceed.
    In addition, if using mutiprocessing, each shard MUST HAVE THE SAME NUMBER OF ENTRIES after filtering
    is applied. We currently do not check for this, but your program may hang if the shards are uneven!

    Notice that a few arguments are different from the AudioLabelDataSet; for example, shuffle (bool) has been
    replaced by shuffle_n (int).

    Additionally, please note that the len() of this DataLayer is assumed to be the length of the manifest
    after filtering. An incorrect manifest length may lead to some DataLoader issues down the line.

    Args:
        audio_tar_filepaths: Either a list of audio tarball filepaths, or a
            string (can be brace-expandable).
        manifest_filepath (str): Path to the manifest.
        labels (list): Dataset parameter.
            List of target classes that can be output by the speaker recognition model.
        featurizer
        shuffle_n (int): How many samples to look ahead and load to be shuffled.
            See WebDataset documentation for more details.
            Defaults to 0.
        min_duration (float): Dataset parameter.
            All training files which have a duration less than min_duration
            are dropped. Note: Duration is read from the manifest JSON.
            Defaults to 0.1.
        max_duration (float): Dataset parameter.
            All training files which have a duration more than max_duration
            are dropped. Note: Duration is read from the manifest JSON.
            Defaults to None.
        trim(bool): Whether to use trim silence from beginning and end
            of audio signal using librosa.effects.trim().
            Defaults to False.
        time_length (float): time length of slice (in seconds) # Pass this only for speaker recognition and VAD task
        shift_length (float): amount of shift of window for generating the frame for VAD task. in a batch # Pass this only for VAD task during inference.
        normalize_audio (bool): Whether to normalize audio signal. Defaults to False.
        shard_strategy (str): Tarred dataset shard distribution strategy chosen as a str value during ddp.
            -   `scatter`: The default shard strategy applied by WebDataset, where each node gets
                a unique set of shards, which are permanently pre-allocated and never changed at runtime.
            -   `replicate`: Optional shard strategy, where each node gets all of the set of shards
                available in the tarred dataset, which are permanently pre-allocated and never changed at runtime.
                The benefit of replication is that it allows each node to sample data points from the entire
                dataset independently of other nodes, and reduces dependence on value of `shuffle_n`.

                Note: Replicated strategy allows every node to sample the entire set of available tarfiles,
                and therefore more than one node may sample the same tarfile, and even sample the same
                data points! As such, there is no assured guarantee that all samples in the dataset will be
                sampled at least once during 1 epoch.
        global_rank (int): Worker rank, used for partitioning shards. Defaults to 0.
        world_size (int): Total number of processes, used for partitioning shards. Defaults to 0.
        is_regression_task (bool): Whether it is a regression task. Defualts to False.
    """

    def __init__(
        self,
        *,
        audio_tar_filepaths: Union[str, List[str]],
        manifest_filepath: Union[str, List[str]],
        labels: List[str],
        featurizer,
        shuffle_n: int = 0,
        min_duration: Optional[float] = 0.1,
        max_duration: Optional[float] = None,
        trim: bool = False,
        shard_strategy: str = "scatter",
        global_rank: int = 0,
        world_size: int = 0,
        is_regression_task: bool = False,
    ):
        self.collection = collections.ASRSpeechLabel(
            manifests_files=manifest_filepath,
            min_duration=min_duration,
            max_duration=max_duration,
            index_by_file_id=True,  # Must set this so the manifest lines can be indexed by file ID
        )

        self.file_occurence = count_occurence(self.collection.mapping)

        self.featurizer = featurizer
        self.trim = trim

        self.labels = labels if labels else self.collection.uniq_labels
        self.num_classes = len(self.labels)

        self.label2id, self.id2label = {}, {}
        for label_id, label in enumerate(self.labels):
            self.label2id[label] = label_id
            self.id2label[label_id] = label

        for idx in range(len(self.labels[:5])):
            logging.debug(" label id {} and its mapped label {}".format(idx, self.id2label[idx]))

        valid_shard_strategies = ['scatter', 'replicate']
        if shard_strategy not in valid_shard_strategies:
            raise ValueError(f"`shard_strategy` must be one of {valid_shard_strategies}")

        if isinstance(audio_tar_filepaths, str):
            # Replace '(' and '[' with '{'
            brace_keys_open = ['(', '[', '<', '_OP_']
            for bkey in brace_keys_open:
                if bkey in audio_tar_filepaths:
                    audio_tar_filepaths = audio_tar_filepaths.replace(bkey, "{")

            # Replace ')' and ']' with '}'
            brace_keys_close = [')', ']', '>', '_CL_']
            for bkey in brace_keys_close:
                if bkey in audio_tar_filepaths:
                    audio_tar_filepaths = audio_tar_filepaths.replace(bkey, "}")

        # Check for distributed and partition shards accordingly
        if world_size > 1:
            if isinstance(audio_tar_filepaths, str):
                # Brace expand
                audio_tar_filepaths = list(braceexpand.braceexpand(audio_tar_filepaths))

            if shard_strategy == 'scatter':
                logging.info("All tarred dataset shards will be scattered evenly across all nodes.")

                if len(audio_tar_filepaths) % world_size != 0:
                    logging.warning(
                        f"Number of shards in tarred dataset ({len(audio_tar_filepaths)}) is not divisible "
                        f"by number of distributed workers ({world_size})."
                    )

                begin_idx = (len(audio_tar_filepaths) // world_size) * global_rank
                end_idx = begin_idx + (len(audio_tar_filepaths) // world_size)
                audio_tar_filepaths = audio_tar_filepaths[begin_idx:end_idx]
                logging.info(
                    "Partitioning tarred dataset: process (%d) taking shards [%d, %d)", global_rank, begin_idx, end_idx
                )

            elif shard_strategy == 'replicate':
                logging.info("All tarred dataset shards will be replicated across all nodes.")

            else:
                raise ValueError(f"Invalid shard strategy ! Allowed values are : {valid_shard_strategies}")

        # Put together WebDataset
        self._dataset = wd.WebDataset(urls=audio_tar_filepaths, nodesplitter=None)

        if shuffle_n > 0:
            self._dataset = self._dataset.shuffle(shuffle_n)
        else:
            logging.info("WebDataset will not shuffle files within the tar files.")

        self._dataset = (
            self._dataset.rename(audio=VALID_FILE_FORMATS, key='__key__')
            .to_tuple('audio', 'key')
            .pipe(self._filter)
            .map(f=self._build_sample)
        )

    def _filter(self, iterator):
        """This function is used to remove samples that have been filtered out by ASRSpeechLabel already.
        Otherwise, we would get a KeyError as _build_sample attempts to find the manifest entry for a sample
        that was filtered out (e.g. for duration).
        Note that if using multi-GPU training, filtering may lead to an imbalance in samples in each shard,
        which may make your code hang as one process will finish before the other.
        """

        class TarredAudioFilter:
            def __init__(self, collection, file_occurence):
                self.iterator = iterator
                self.collection = collection
                self.file_occurence = file_occurence
                self._iterable = None

            def __iter__(self):
                self._iterable = self._internal_generator()
                return self

            def __next__(self):
                try:
                    values = next(self._iterable)
                except StopIteration:
                    # reset generator
                    self._iterable = self._internal_generator()
                    values = next(self._iterable)

                return values

            def _internal_generator(self):
                """
                WebDataset requires an Iterator, but we require an iterable that yields 1-or-more
                values per value inside self.iterator.

                Therefore wrap the iterator with a generator function that will yield 1-or-more
                values per sample in the iterator.
                """
                for _, tup in enumerate(self.iterator):
                    audio_bytes, audio_filename = tup

                    file_id, _ = os.path.splitext(os.path.basename(audio_filename))
                    if audio_filename in self.file_occurence:
                        for j in range(0, self.file_occurence[file_id]):
                            if j == 0:
                                audio_filename = file_id
                            else:
                                audio_filename = file_id + "-sub" + str(j)
                            yield audio_bytes, audio_filename

        return TarredAudioFilter(self.collection, self.file_occurence)

    def _build_sample(self, tup):
        """Builds the training sample by combining the data from the WebDataset with the manifest info.
        """
        audio_bytes, audio_filename = tup
        # Grab manifest entry from self.collection
        file_id, _ = os.path.splitext(os.path.basename(audio_filename))

        manifest_idx = self.collection.mapping[file_id]
        manifest_entry = self.collection[manifest_idx]

        offset = manifest_entry.offset
        if offset is None:
            offset = 0

        # Convert audio bytes to IO stream for processing (for SoundFile to read)
        audio_filestream = io.BytesIO(audio_bytes)
        features = self.featurizer.process(
            audio_filestream, offset=offset, duration=manifest_entry.duration, trim=self.trim,
        )

        audio_filestream.close()

        # Audio features
        f, fl = features, torch.tensor(features.shape[0]).long()

        t = self.label2id[manifest_entry.label]
        tl = 1  # For compatibility with collate_fn used later

        return f, fl, torch.tensor(t).long(), torch.tensor(tl).long()

    def __iter__(self):
        return self._dataset.__iter__()

    def __len__(self):
        return len(self.collection)


class TarredAudioToClassificationLabelDataset(_TarredAudioLabelDataset):
    """
    A similar Dataset to the AudioToClassificationLabelDataset, but which loads tarred audio files.

    Accepts a single comma-separated JSON manifest file (in the same style as for the AudioToClassificationLabelDataset),
    as well as the path(s) to the tarball(s) containing the wav files. Each line of the manifest should
    contain the information for one audio file, including at least the transcript and name of the audio
    file within the tarball.

    Valid formats for the audio_tar_filepaths argument include:
    (1) a single string that can be brace-expanded, e.g. 'path/to/audio.tar' or 'path/to/audio_{1..100}.tar.gz', or
    (2) a list of file paths that will not be brace-expanded, e.g. ['audio_1.tar', 'audio_2.tar', ...].

    See the WebDataset documentation for more information about accepted data and input formats.

    If using multiple processes the number of shards should be divisible by the number of workers to ensure an
    even split among workers. If it is not divisible, logging will give a warning but training will proceed.
    In addition, if using mutiprocessing, each shard MUST HAVE THE SAME NUMBER OF ENTRIES after filtering
    is applied. We currently do not check for this, but your program may hang if the shards are uneven!

    Notice that a few arguments are different from the AudioToBPEDataset; for example, shuffle (bool) has been
    replaced by shuffle_n (int).

    Additionally, please note that the len() of this DataLayer is assumed to be the length of the manifest
    after filtering. An incorrect manifest length may lead to some DataLoader issues down the line.

    Args:
        audio_tar_filepaths: Either a list of audio tarball filepaths, or a
            string (can be brace-expandable).
        manifest_filepath (str): Path to the manifest.
        labels (list): Dataset parameter.
            List of target classes that can be output by the speaker recognition model.
        featurizer
        shuffle_n (int): How many samples to look ahead and load to be shuffled.
            See WebDataset documentation for more details.
            Defaults to 0.
        min_duration (float): Dataset parameter.
            All training files which have a duration less than min_duration
            are dropped. Note: Duration is read from the manifest JSON.
            Defaults to 0.1.
        max_duration (float): Dataset parameter.
            All training files which have a duration more than max_duration
            are dropped. Note: Duration is read from the manifest JSON.
            Defaults to None.
        trim(bool): Whether to use trim silence from beginning and end
            of audio signal using librosa.effects.trim().
            Defaults to False.
        shard_strategy (str): Tarred dataset shard distribution strategy chosen as a str value during ddp.
            -   `scatter`: The default shard strategy applied by WebDataset, where each node gets
                a unique set of shards, which are permanently pre-allocated and never changed at runtime.
            -   `replicate`: Optional shard strategy, where each node gets all of the set of shards
                available in the tarred dataset, which are permanently pre-allocated and never changed at runtime.
                The benefit of replication is that it allows each node to sample data points from the entire
                dataset independently of other nodes, and reduces dependence on value of `shuffle_n`.

                Note: Replicated strategy allows every node to sample the entire set of available tarfiles,
                and therefore more than one node may sample the same tarfile, and even sample the same
                data points! As such, there is no assured guarantee that all samples in the dataset will be
                sampled at least once during 1 epoch.
        global_rank (int): Worker rank, used for partitioning shards. Defaults to 0.
        world_size (int): Total number of processes, used for partitioning shards. Defaults to 0.
        is_regression_task (bool): Whether it is a regression task. Defualts to False.
    """

    # self.labels = labels if labels else self.collection.uniq_labels
    # self.num_commands = len(self.labels)

    def _collate_fn(self, batch):
        return _speech_collate_fn(batch, pad_id=0)


class TarredAudioToSpeechLabelDataset(_TarredAudioLabelDataset):
    """
    A similar Dataset to the AudioToSpeechLabelDataset, but which loads tarred audio files.

    Accepts a single comma-separated JSON manifest file (in the same style as for the AudioToSpeechLabelDataset),
    as well as the path(s) to the tarball(s) containing the wav files. Each line of the manifest should
    contain the information for one audio file, including at least the transcript and name of the audio
    file within the tarball.

    Valid formats for the audio_tar_filepaths argument include:
    (1) a single string that can be brace-expanded, e.g. 'path/to/audio.tar' or 'path/to/audio_{1..100}.tar.gz', or
    (2) a list of file paths that will not be brace-expanded, e.g. ['audio_1.tar', 'audio_2.tar', ...].

    See the WebDataset documentation for more information about accepted data and input formats.

    If using multiple processes the number of shards should be divisible by the number of workers to ensure an
    even split among workers. If it is not divisible, logging will give a warning but training will proceed.
    In addition, if using mutiprocessing, each shard MUST HAVE THE SAME NUMBER OF ENTRIES after filtering
    is applied. We currently do not check for this, but your program may hang if the shards are uneven!

    Notice that a few arguments are different from the AudioToBPEDataset; for example, shuffle (bool) has been
    replaced by shuffle_n (int).

    Additionally, please note that the len() of this DataLayer is assumed to be the length of the manifest
    after filtering. An incorrect manifest length may lead to some DataLoader issues down the line.

    Args:
        audio_tar_filepaths: Either a list of audio tarball filepaths, or a
            string (can be brace-expandable).
        manifest_filepath (str): Path to the manifest.
        labels (list): Dataset parameter.
            List of target classes that can be output by the speaker recognition model.
        featurizer
        shuffle_n (int): How many samples to look ahead and load to be shuffled.
            See WebDataset documentation for more details.
            Defaults to 0.
        min_duration (float): Dataset parameter.
            All training files which have a duration less than min_duration
            are dropped. Note: Duration is read from the manifest JSON.
            Defaults to 0.1.
        max_duration (float): Dataset parameter.
            All training files which have a duration more than max_duration
            are dropped. Note: Duration is read from the manifest JSON.
            Defaults to None.
        trim(bool): Whether to use trim silence from beginning and end
            of audio signal using librosa.effects.trim().
            Defaults to False.
        time_length (float): time length of slice (in seconds) # Pass this only for speaker recognition and VAD task
        shift_length (float): amount of shift of window for generating the frame for VAD task. in a batch # Pass this only for VAD task during inference.
        normalize_audio (bool): Whether to normalize audio signal. Defaults to False.
        shard_strategy (str): Tarred dataset shard distribution strategy chosen as a str value during ddp.
            -   `scatter`: The default shard strategy applied by WebDataset, where each node gets
                a unique set of shards, which are permanently pre-allocated and never changed at runtime.
            -   `replicate`: Optional shard strategy, where each node gets all of the set of shards
                available in the tarred dataset, which are permanently pre-allocated and never changed at runtime.
                The benefit of replication is that it allows each node to sample data points from the entire
                dataset independently of other nodes, and reduces dependence on value of `shuffle_n`.

                Note: Replicated strategy allows every node to sample the entire set of available tarfiles,
                and therefore more than one node may sample the same tarfile, and even sample the same
                data points! As such, there is no assured guarantee that all samples in the dataset will be
                sampled at least once during 1 epoch.
        global_rank (int): Worker rank, used for partitioning shards. Defaults to 0.
        world_size (int): Total number of processes, used for partitioning shards. Defaults to 0.
    """

    def __init__(
        self,
        *,
        audio_tar_filepaths: Union[str, List[str]],
        manifest_filepath: Union[str, List[str]],
        labels: List[str],
        featurizer,
        shuffle_n: int = 0,
        min_duration: Optional[float] = 0.1,
        max_duration: Optional[float] = None,
        trim: bool = False,
        time_length: Optional[float] = 8,
        shift_length: Optional[float] = 1,
        normalize_audio: bool = False,
        shard_strategy: str = "scatter",
        global_rank: int = 0,
        world_size: int = 0,
    ):
        logging.info("Time length considered for collate func is {}".format(time_length))
        logging.info("Shift length considered for collate func is {}".format(shift_length))
        self.time_length = time_length
        self.shift_length = shift_length
        self.normalize_audio = normalize_audio

        super().__init__(
            audio_tar_filepaths=audio_tar_filepaths,
            manifest_filepath=manifest_filepath,
            labels=labels,
            featurizer=featurizer,
            shuffle_n=shuffle_n,
            min_duration=min_duration,
            max_duration=max_duration,
            trim=trim,
            shard_strategy=shard_strategy,
            global_rank=global_rank,
            world_size=world_size,
        )

    def fixed_seq_collate_fn(self, batch):
        return _fixed_seq_collate_fn(self, batch)

    def sliced_seq_collate_fn(self, batch):
        return _sliced_seq_collate_fn(self, batch)

    def vad_frame_seq_collate_fn(self, batch):
        return _vad_frame_seq_collate_fn(self, batch)
