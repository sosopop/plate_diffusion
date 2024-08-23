
class LicensePlateVocab:
    pad_token='#'
    eos_token='$'
    bos_token='^'
    max_length = 16
    vocab_list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '云', '京', '冀', '吉', '学', '宁', '川', '挂', '新', '晋', '桂', '沪', '津', '浙', '渝', '湘', '琼', '甘', '皖', '粤', '苏', '蒙', '藏', '警', '豫', '贵', '赣', '辽', '鄂', '闽', '陕', '青', '鲁', '黑', pad_token, eos_token, bos_token]
    vocab_dict = {char: idx for idx, char in enumerate(vocab_list)}
    idx_dict = {idx: char for idx, char in enumerate(vocab_list)}
    pad_idx = vocab_dict[pad_token]
    eos_idx = vocab_dict[eos_token]
    bos_idx = vocab_dict[bos_token]
    vocab_size = len(vocab_list)
    
    @staticmethod
    def text_to_sequence(text, max_length=max_length, pad_to_max_length=True, add_eos=False, add_bos=False):
        sequence = []
        if add_bos:
            sequence.append(LicensePlateVocab.bos_idx)  # Add BOS token at the beginning
        for char in text:
            if char in LicensePlateVocab.vocab_dict:
                sequence.append(LicensePlateVocab.vocab_dict[char])
        if add_eos:
            sequence.append(LicensePlateVocab.eos_idx)  # Add EOS token at the end
        if len(sequence) < max_length:
            if pad_to_max_length:
                sequence = sequence + [LicensePlateVocab.pad_idx] * (max_length - len(sequence))
        else:
            sequence = sequence[:max_length]
        return sequence

    @staticmethod
    def sequence_to_text( sequence):
        return ''.join([LicensePlateVocab.idx_dict[idx] for idx in sequence if idx != LicensePlateVocab.pad_idx and idx != LicensePlateVocab.eos_idx and idx != LicensePlateVocab.bos_idx])
