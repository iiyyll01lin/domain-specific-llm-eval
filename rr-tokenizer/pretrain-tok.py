from tokenizers import Tokenizer, models, trainers, pre_tokenizers

tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
trainer = trainers.BpeTrainer(vocab_size=10000, min_frequency=2, special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"])

files = ["/ocp_redfish_dataset.txt"]
tokenizer.train(files, trainer)
tokenizer.save("ocp_redfish_tokenizer.json")
tokenizer = Tokenizer.from_file("ocp_redfish_tokenizer.json")
