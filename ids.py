import argparse
from kokoro import KModel, KPipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser("convert text to input ids.", add_help=True)
    parser.add_argument("--text", "-t", type=str, required=True, help="text")
    parser.add_argument("--config_file", "-c", type=str, default="checkpoints/config.json", help="path to config file")
    parser.add_argument(
        "--checkpoint_path", "-p", type=str, default="checkpoints/kokoro-v1_0.pth", help="path to checkpoint file"
    )

    args = parser.parse_args()
    config_file = args.config_file  # change the path of the model config file
    checkpoint_path = args.checkpoint_path  # change the path of the model
    text = args.text

    model = KModel(config_file, checkpoint_path)
    pipeline = KPipeline(lang_code='a', model=model, device='cpu')

    _, tokens = pipeline.g2p(text)
    for gs, ps, tks in pipeline.en_tokenize(tokens):
        if not ps:
            continue

    if len(ps) > 510:
        ps = ps[:510]

    print(f'gs: {gs} -> ps: {ps} -> tokens: {tks}')

    input_ids = list(filter(lambda i: i is not None, map(lambda p: model.vocab.get(p), ps)))
    print(f"text: {text} -> phonemes: {ps} -> input_ids: {input_ids}")
