import ctc_segmentation as cs
import numpy as np

from nemo.collections import asr as nemo_asr


def prepare_tokenized_text_nemo(text, asr_model="stt_en_citrinet_512_gamma_0_25"):
    asr_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(asr_model)
    tokenizer = asr_model.tokenizer
    space_idx = vocabulary.index("▁")
    blank_idx = len(vocabulary) - 1

    ground_truth_mat = [[-1]]
    utt_begin_indices = []
    for uttr in text:
        ground_truth_mat += [[space_idx]]
        utt_begin_indices.append(len(ground_truth_mat))
        token_ids = tokenizer.text_to_ids(uttr)
        uttr_ids = []
        for id in token_ids:
            uttr_ids.append([id])
            # uttr_ids.append([space_idx])
        ground_truth_mat += uttr_ids  # [[t] for t in token_ids]
        ground_truth_mat += [[space_idx]]

    utt_begin_indices.append(len(ground_truth_mat))
    ground_truth_mat += [[space_idx]]

    ground_truth_mat = np.array(ground_truth_mat, np.int64)
    return ground_truth_mat, utt_begin_indices, vocabulary


def prepare_tokenized_text_nemo_works(text, asr_model):
    """ WORKS DO NOT CHANGE"""
    try:
        asr_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(asr_model)
    except:
        asr_model = nemo_asr.models.EncDecCTCModelBPE.restore_from(asr_model)
    vocabulary = list(asr_model.cfg.decoder.vocabulary) + ["ε"]
    tokenizer = asr_model.tokenizer
    space_idx = len(vocabulary) - 1  # vocabulary.index("▁")

    ground_truth_mat = [[-1]]
    utt_begin_indices = []
    for uttr in text:
        ground_truth_mat += [[space_idx]]
        utt_begin_indices.append(len(ground_truth_mat))
        token_ids = tokenizer.text_to_ids(uttr)
        ground_truth_mat += [[t] for t in token_ids]

    utt_begin_indices.append(len(ground_truth_mat))
    ground_truth_mat += [[space_idx]]

    ground_truth_mat = np.array(ground_truth_mat, np.int64)
    return ground_truth_mat, utt_begin_indices, vocabulary


def prepare_tokenized_text_nemo_works_modified(text, tokenizer, vocabulary):
    """ WIP """
    space_idx = vocabulary.index("▁")
    blank_idx = len(vocabulary) - 1

    ground_truth_mat = [[-1, -1]]
    utt_begin_indices = []
    for uttr in text:
        ground_truth_mat += [[blank_idx, space_idx]]
        utt_begin_indices.append(len(ground_truth_mat))
        token_ids = tokenizer.text_to_ids(uttr)
        ground_truth_mat += [[t, -1] for t in token_ids]

    utt_begin_indices.append(len(ground_truth_mat))
    ground_truth_mat += [[blank_idx, space_idx]]
    # print(ground_truth_mat)
    ground_truth_mat = np.array(ground_truth_mat, np.int64)
    return ground_truth_mat, utt_begin_indices


def prepare_tokenized_text(text, vocabulary):
    """Prepare the given tokenized text for CTC segmentation.
    :param config: an instance of CtcSegmentationParameters
    :param text: string with tokens separated by spaces
    :return: label matrix, character index matrix
    """
    config = cs.CtcSegmentationParameters()
    config.char_list = vocabulary
    config.index_duration = 0.0799983368347339
    config.blank = len(vocabulary) - 1
    config.space = "▁"
    # config.replace_spaces_with_blanks=True
    # config.tokenized_meta_symbol = "▁"

    ground_truth = [config.start_of_ground_truth]
    utt_begin_indices = []
    for utt in text:
        # One space in-between
        if not ground_truth[-1] == config.space:
            ground_truth += [config.space]
        # Start new utterance remember index
        utt_begin_indices.append(len(ground_truth) - 1)
        # Add tokens of utterance
        for token in utt.split():
            if token in config.char_list:
                # import pdb; pdb.set_trace()
                if config.replace_spaces_with_blanks and not token.startswith(config.tokenized_meta_symbol):
                    ground_truth += [config.space]
                ground_truth += [token]
    # Add space to the end
    if not ground_truth[-1] == config.space:
        ground_truth += [config.space]
    print(f"ground_truth: {ground_truth}")
    utt_begin_indices.append(len(ground_truth) - 1)
    # Create matrix: time frame x number of letters the character symbol spans
    max_char_len = 1
    ground_truth_mat = np.ones([len(ground_truth), max_char_len], np.int64) * -1
    for i in range(1, len(ground_truth)):
        if ground_truth[i] == config.space:
            ground_truth_mat[i, 0] = config.blank
        else:
            char_index = config.char_list.index(ground_truth[i])
            ground_truth_mat[i, 0] = char_index
    return ground_truth_mat, utt_begin_indices


def _print(ground_truth_mat, vocabulary):
    chars = []
    for row in ground_truth_mat:
        chars.append([])
        for ch_id in row:
            if ch_id != -1:
                chars[-1].append(vocabulary[int(ch_id)])

    [print(x) for x in chars]


def get_words(fpath, tokenizer):
    words = []
    with open(fpath, 'r') as f:
        for line in f:
            words.extend(line.strip().lower().split())
    # import pdb; pdb.set_trace()
    words = [" ".join(tokenizer.text_to_tokens(w)) for w in words]
    words = [w.replace("▁", "") for w in words]
    print(words[:5])
    with open("/home/ebakhturina/data/segmentation/test/processed/1.txt", "w") as f_out:
        f_out.write("\n".join(words))
    return words


def prepare_text_default(config, text, qn=True, char_list=None):
    # temporary compatibility fix for previous espnet versions

    config.replace_spaces_with_blanks = True
    if type(config.blank) == str:
        config.blank = 0
    if char_list is not None:
        config.char_list = char_list
    blank = config.char_list[config.blank]
    ground_truth = config.start_of_ground_truth
    utt_begin_indices = []
    for utt in text:
        # One space in-between
        if not ground_truth.endswith(config.space):
            ground_truth += config.space
        # Start new utterance remember index
        utt_begin_indices.append(len(ground_truth) - 1)
        # Add chars of utterance
        for char in utt:
            # if char.isspace() and config.replace_spaces_with_blanks:
            #     if not ground_truth.endswith(config.space):
            #         ground_truth += config.space
            if char in config.char_list and char not in config.excluded_characters:
                ground_truth += char
            elif config.tokenized_meta_symbol + char in config.char_list:
                ground_truth += char
    # Add space to the end
    if not ground_truth.endswith(config.space):
        ground_truth += config.space
    print(f"ground_truth: {ground_truth}")
    utt_begin_indices.append(len(ground_truth) - 1)
    # Create matrix: time frame x number of letters the character symbol spans
    # for QN:
    if qn:
        max_char_len = 2
    else:
        max_char_len = max([len(c) for c in config.char_list])

    ground_truth_mat = np.ones([len(ground_truth), max_char_len], np.int64) * -1
    for i in range(len(ground_truth)):
        for s in range(max_char_len):
            if i - s < 0:
                continue
            span = ground_truth[i - s : i + 1]
            # print(f'{i} -- {span} --> {span.replace(config.space, blank)}')
            # if 'ri' in span:
            # print(span)
            if span == config.space:
                if not qn:
                    span = span.replace(config.space, blank)
                    char_index = config.char_list.index(span)
                    ground_truth_mat[i, s] = char_index
                    ground_truth_mat[i, s + 1] = config.char_list.index(config.space)
                else:
                    char_index = config.char_list.index(span)
                    ground_truth_mat[i, s] = char_index
                    # import pdb; pdb.set_trace()
                    # if (i+1) not in utt_begin_indices:
                    #     ground_truth_mat[i, s + 1] = config.blank
                # import pdb; pdb.set_trace()
                # print()
            # if span in config.char_list:
            #     char_index = config.char_list.index(span)
            #     ground_truth_mat[i, s] = char_index
            if not qn:
                if not span.startswith(config.space) and (config.tokenized_meta_symbol + span) in config.char_list:
                    char_index = config.char_list.index(config.tokenized_meta_symbol + span)
                    ground_truth_mat[i, s] = char_index
                elif span.startswith(config.space) and span[1:] in config.char_list:
                    import pdb

                    pdb.set_trace()
                    char_index = config.char_list.index(span[1:])
                    ground_truth_mat[i, s] = char_index
            elif qn and span in config.char_list:
                char_index = config.char_list.index(span)
                ground_truth_mat[i, s] = char_index
    return ground_truth_mat, utt_begin_indices


def get_config():
    asr_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained("stt_en_citrinet_512_gamma_0_25")
    vocabulary = list(asr_model.cfg.decoder.vocabulary) + ["ε"]
    tokenizer = asr_model.tokenizer

    config = cs.CtcSegmentationParameters()
    config.char_list = vocabulary
    config.blank = len(vocabulary) - 1
    config.space = "▁"
    config.tokenized_meta_symbol = "##"
    return config, tokenizer


def get_config_match_cs():
    asr_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained("stt_en_citrinet_512_gamma_0_25")
    vocabulary = list(asr_model.cfg.decoder.vocabulary) + ["ε"]
    # tokenizer = asr_model.tokenizer
    for i in range(len(vocabulary) - 1):
        if not vocabulary[i].startswith("##"):
            vocabulary[i] = "▁" + vocabulary[i]
        else:
            vocabulary[i] = vocabulary[i].replace("##", "")

    config = cs.CtcSegmentationParameters()
    config.char_list = vocabulary
    config.blank = len(vocabulary) - 1
    # config.space = "▁"
    # config.tokenized_meta_symbol = "##"
    return config


def get_config_qn():
    asr_model = nemo_asr.models.EncDecCTCModel.from_pretrained("QuartzNet15x5Base-En")
    vocabulary = list(asr_model.cfg.decoder.vocabulary) + ["ε"]
    config = cs.CtcSegmentationParameters()
    config.char_list = vocabulary
    config.blank = len(vocabulary) - 1
    config.space = " "
    return config


if __name__ == "__main__":
    """
    WORKS
    text = ["a carrier", "upon"]
    ground_truth_mat, utt_begin_indices, vocabulary = prepare_tokenized_text_nemo_works_modified(
        text, "stt_en_citrinet_512_gamma_0_25"
    )
    _print(ground_truth_mat, vocabulary)
    print('\n')
    print('-' * 40)
    import pdb
    pdb.set_trace()
    """

    # QN
    text = ["a carrier", "upon"]
    # config.tokenized_meta_symbol = "##"
    config = get_config_qn()
    ground_truth_mat, utt_begin_indices = prepare_text_default(config, text)
    _print(ground_truth_mat, config.char_list)

    """
    text = ["a carrier", "upon market"]
    for i in range(len(text)):
        text[i] = " ".join(["▁" + x for x in text[i].split()])
    print(text)
    config, tokenizer = get_config()
    ground_truth_mat, utt_begin_indices = prepare_text_default(config, text)
    _print(ground_truth_mat, config.char_list)
    print('\n')
    print('-' * 40)
    import pdb;

    pdb.set_trace()
    print()
    """
    print()


"""
[]
['<unk>']
['▁']
['r', '▁r']
['e', 're', '▁re']
['a']
['l', 'al', '▁real']
['l', 'll', 'all']
['y', 'ly', 'ally', '▁really']
['▁']
['l', '▁l']
['i', '▁li']
['k']
['e', 'ke', '▁like']
['▁']
['t', '▁t']
['e', '▁te']
['c']
['h', 'ch']
['n']
['o']
['l', 'ol']
['o']
['g', 'og', 'olog']
['y']
['<unk>']
['h']
['e']
['▁']
['q']
['u', 'qu', '▁qu']



"""
