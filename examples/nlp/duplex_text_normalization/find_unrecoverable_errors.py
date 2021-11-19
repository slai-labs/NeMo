# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

import re
import string
import sys

from tqdm import tqdm

try:
    from nemo_text_processing.text_normalization.normalize_with_audio import NormalizerWithAudio

    PYNINI_AVAILABLE = True
except (ModuleNotFoundError, ImportError):
    PYNINI_AVAILABLE = False

input_file = sys.argv[
    1
]  # "/home/yzhang/code/NeMo/examples/nlp/duplex_text_normalization/errors_no_5_digit_electronic.txt"

normalizer = NormalizerWithAudio(
    input_case='cased', lang='en', cache_dir="/home/ebakhturina/NeMo/nemo_text_processing/text_normalization/cache_dir"
)

print('Processing: ', input_file)


def remove_punctuation(word: str, remove_spaces=True, do_lower=True):
    """
    Removes all punctuation marks from a word except for '
    that is often a part of word: don't, it's, and so on
    """
    all_punct_marks = string.punctuation
    word = re.sub('[' + all_punct_marks + ']', '', word)

    if remove_spaces:
        word = word.replace(" ", "").replace(u"\u00A0", "").strip()

    if do_lower:
        word = word.lower()
    return word


original_wrong = 0
wrong = 0
correct_with_no_punct = 0
correct_with_no_sil = 0
correct_with_zs = 0
acceptable_error = 0

example = []
with open(input_file.replace(".txt", "_reduced.txt"), 'w') as f_out:
    with open(input_file, 'r') as f:
        for line in tqdm(f):
            if not line:
                continue
            example.append(line[::-1][: line[::-1].find('\t')].strip()[::-1])
            if len(example) == 4:
                original_wrong += 1
                _input_full, _input, pred, target = example
                pred_no_punct = remove_punctuation(pred).strip()
                target_no_punct = remove_punctuation(target).strip()
                if pred_no_punct == target_no_punct:
                    correct_with_no_punct += 1
                elif pred_no_punct.replace(" sil ", " ") == target_no_punct:
                    correct_with_no_sil += 1
                elif pred_no_punct.replace("s", "z") == target_no_punct.replace("s", "z"):
                    correct_with_zs += 1
                else:
                    wfst_pred = normalizer.normalize(_input.replace("``", "").strip(), n_tagged=100000)
                    wfst_pred = [remove_punctuation(x) for x in wfst_pred]

                    if pred_no_punct in wfst_pred:
                        acceptable_error += 1
                    else:
                        # print('input: ', _input)
                        # print('nn...:', pred)
                        # print('target:', target)
                        # print("-" * 40)
                        f_out.write(line[: line.find('\t')].strip() + '\t' + "INPUT" + '\t' + example[0] + '\n')
                        f_out.write(line[: line.find('\t')].strip() + '\t' + "Cente" + '\t' + example[1] + '\n')
                        f_out.write(line[: line.find('\t')].strip() + '\t' + "PREDI" + '\t' + example[2] + '\n')
                        f_out.write(line[: line.find('\t')].strip() + '\t' + "TARGE" + '\t' + example[3] + '\n')
                        f_out.write("-" * 40 + "\n")
                        wrong += 1

                example = []

                # print(f"acceptable_error {acceptable_error}, nonacc {wrong}")
print(f'original wrong: {original_wrong}')
print(f'wrong: {wrong}')
print(f'no punct correct: {correct_with_no_punct}')
print(f'no sil correct: {correct_with_no_punct}')
print(f's z correct: {correct_with_zs}')
print(f'Acceptable error: {acceptable_error}')
