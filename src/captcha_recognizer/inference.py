import pickle

import torch

import config


class CTCDecoder:
    def __init__(
        self,
        decoder_path: str = str(config.DICT_PATH / "decode_dict.pkl"),
        blank_idx: int = 0,
    ):
        self.decoder_path = decoder_path
        self.blank_idx = blank_idx
        with open(self.decoder_path, "rb") as f:
            self.decode_dict = pickle.load(f)
        self.blank_token = self.decode_dict[blank_idx]

    def decode(self, log_prob_input):
        # input: (bs, seq_length, no_tokens)
        preds_idx_all = torch.argmax(
            log_prob_input, dim=2
        )  # bs, seq_length (tensor[idx, idx, idx ...])

        preds_idx = []
        preds_all = []
        preds = []

        for pred_idx in preds_idx_all:
            pred_collapsed = torch.unique_consecutive(pred_idx).detach().cpu().numpy()
            preds_idx.append(pred_collapsed)
            pred_sequence_all = "".join(
                [self.decode_dict[idx.item()] for idx in pred_idx]
            )
            preds_all.append(pred_sequence_all)
            pred_sequence = "".join(
                [
                    self.decode_dict[idx]
                    for idx in pred_collapsed
                    if self.decode_dict[idx] != self.blank_token
                ]
            )
            preds.append(pred_sequence)

        return preds_idx, preds_all, preds


if __name__ == "__main__":
    decoder = CTCDecoder()
    probs = torch.rand(5, 75, len(decoder.decode_dict))
    log_probs = torch.softmax(probs, dim=2)
    print(decoder.decode(log_probs))
