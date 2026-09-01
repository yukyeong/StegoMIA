import os
import argparse
import logging
import pandas as pd
from tqdm import tqdm
from utils import config
from .eda import *

_WORDNET_MISSING = False

def _augment_text(caption):
    global _WORDNET_MISSING
    # Handle empty or invalid captions
    if not caption or not isinstance(caption, str) or len(caption.strip()) == 0:
        return caption
    
    try:
        augmented_caption = eda(caption)
        return augmented_caption[0] if augmented_caption else caption
    except (LookupError, ValueError, IndexError) as e:
        if isinstance(e, LookupError) and not _WORDNET_MISSING:
            logging.warning(
                "NLTK wordnet not available; falling back to original captions for text augmentation."
            )
            _WORDNET_MISSING = True
        # Return original caption if augmentation fails
        return caption

def augment_text(options):
    df = pd.read_csv(os.path.join(config.root, options.input_file), delimiter = options.delimiter)
    captions = df[options.caption_key]

    augmented_captions = []
    for caption in tqdm(captions):
        augmented_caption = eda(caption)
        augmented_captions.append(augmented_caption[0])
    
    df["augmented_" + options.caption_key] = augmented_captions
    df.to_csv(os.path.join(config.root, options.output_file), index = False)

if(__name__ == "__main__"):
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-i,--input_file", dest = "input_file", type = str, required = True, help = "Input file")
    parser.add_argument("-o,--output_file", dest = "output_file", type = str, required = True, help = "Output file")
    parser.add_argument("--delimiter", type = str, default = ",", help = "Input file delimiter")
    parser.add_argument("--caption_key", type = str, default = "caption", help = "Caption column name")

    options = parser.parse_args()
    augment_text(options)
