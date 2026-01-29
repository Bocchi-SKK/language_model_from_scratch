from cs336_basics.run_scripts.filepath import *
from cs336_basics import train_bpe

vocab,merges = train_bpe.train_bpe(input_path=TS_TRAIN_PATH, vocab_size=10000, special_tokens=SPECIAL_TOKENS)
train_bpe.save_vocab(vocab=vocab, save_path=TS_VOCAB_PATH)
train_bpe.save_merges(merges=merges, save_path=TS_MERGES_PATH)

vocab,merges = train_bpe.train_bpe(input_path=OWT_TRAIN_PATH, vocab_size=32000, special_tokens=SPECIAL_TOKENS)
train_bpe.save_vocab(vocab=vocab, save_path=OWT_VOCAB_PATH)
train_bpe.save_merges(merges=merges, save_path=OWT_MERGES_PATH)