# main.py

import os
import argparse
import logging
from datetime import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm

from data_processing import DataProcessor       
from embedding_loader import FastTextEmbeddings, TransformerStaticEmbeddings
from concreteness_predictor import ConcretenessPredictor
from evaluation import Evaluator                 

# Logging to both console and log.txt 
logger = logging.getLogger()
logger.setLevel(logging.INFO)
fh = logging.FileHandler("log.txt", mode="a")
fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(fh)

EMBEDDINGS_ROOT = "embeddings"

def _derive_embedding_cfg(args, default_cfg):
    """
    Construct embedding configuration for experiment, allowing for CLI/user overrides.
    
    Args:
        args: Namespace from argparse with CLI/user overrides.
        default_cfg: Dictionary of default parameters for the language.
    
    Returns:
        cfg: Dictionary with all necessary settings for loading embeddings and running the experiment.
    """
    cfg = default_cfg.copy()
    emb_dir = os.path.join(EMBEDDINGS_ROOT, args.language)
    os.makedirs(emb_dir, exist_ok=True)
    # Override k/weighting if provided via CLI
    if args.k is not None:
        cfg['k'] = args.k
    if args.weighting is not None:
        cfg['weighting'] = args.weighting

    # Embedding type override
    if args.embedding_type == 'transformer':
        cfg['emb_type'] = 'transformer'
        # Select static embedding file for transformer, using model variant and mode
        if args.model_path is None:
            base = args.model_variant.lower()
            mode = 'para' if args.static_mode == 'para' else 'sent'
            fname = f"{base}_{24 if base != 'gpt2' else '24'}layer_{mode}.vec"
        else:
            fname = args.model_path if os.path.isabs(args.model_path) else args.model_path
        cfg['emb_path'] = (fname if os.path.isabs(fname) else os.path.join(emb_dir, fname))
        cfg['emb_fmt']  = 'vec'
    else:
        # fastText configuration
        cfg['emb_type'] = 'fasttext'
        if args.model_path is None:
            # Use per-language default filename
            cfg['emb_path'] = os.path.join(emb_dir, default_cfg['emb_path'])
        else:
            mp = args.model_path
            cfg['emb_path'] = mp if os.path.isabs(mp) else os.path.join(emb_dir, mp)
        # file_format override
        if args.file_format is not None:
            cfg['emb_fmt'] = args.file_format

    return cfg


def _load_embeddings(cfg):
    """
    Load embeddings according to configuration dictionary.
    Supports both fastText and transformer static embeddings.
    """
    if cfg['emb_type'] == 'fasttext':
        emb = FastTextEmbeddings(model_path=cfg['emb_path'],
                                 file_format=cfg['emb_fmt'])
    else:
        emb = TransformerStaticEmbeddings(model_path=cfg['emb_path'])
    emb.load_embeddings()
    return emb


def run_experiment_sweep(lang, data_file, test_size, n_folds, default_cfg, sweep_name, param_list):
    """
    Run a hyperparameter sweep over parameter grid.
    Each config in param_list is evaluated with k-fold CV; results are logged per fold and averaged.
    
    Args:
        lang: 'en' or 'ar'.
        data_file: Path to main dataset file.
        test_size: Proportion for test split.
        n_folds: Number of cross-validation folds.
        default_cfg: Default config dictionary.
        sweep_name: Name for output files.
        param_list: List of parameter dictionaries (each config is tested).
    """
    # Prepare train split and k-folds
    dp = DataProcessor(file_path=data_file,
                       word_column='word',
                       score_column='concreteness',
                       train_file=f"{lang}_training.csv",
                       test_file =f"{lang}_test.csv")
    train_set, _ = dp.split_data(test_size=test_size, random_state=42, stratify=False)
    dp.train_set = train_set
    folds = list(dp.create_folds(n_splits=n_folds, random_state=42, stratify=False))

    records = []
    for cfg in tqdm(param_list, desc=sweep_name):
        # Load embeddings and initialize predictor/evaluator
        emb = _load_embeddings(cfg)
        pred = ConcretenessPredictor(embeddings=emb,
                                     k=cfg['k'],
                                     weighting=cfg['weighting'])
        ev = Evaluator(predictor=pred)

        # Evaluate on each fold
        fold_metrics = []
        for fold_idx, (tr, va) in enumerate(folds, start=1):
            tr_df = train_set.iloc[tr]
            va_df = train_set.iloc[va]

            pred.fit(tr_df)
            y_pred = pred.predict(va_df)
            m = ev.compute_metrics(y_true=va_df['concreteness'].values,
                                   y_pred=y_pred)

            rec = {
                **cfg,
                'fold': fold_idx,
                'rmse':          m['rmse'],
                'pearson_corr':  m['pearson_corr'],
                'spearman_corr': m['spearman_corr'],
                            }
            records.append(rec)
            fold_metrics.append(m)

        # Add row summarizing average of all folds for this parameter setting
        avg = {
            'fold': 'all',
            'rmse':          np.mean([m['rmse']          for m in fold_metrics]),
            'pearson_corr':  np.mean([m['pearson_corr']  for m in fold_metrics]),
            'spearman_corr': np.mean([m['spearman_corr'] for m in fold_metrics]),
        }
        records.append({**cfg, **avg})

    # Write results to disk
    result_dir = os.path.join("results", lang)
    os.makedirs(result_dir, exist_ok=True)
    out = os.path.join(result_dir, f"results_{lang}_{sweep_name}.csv")
    pd.DataFrame(records).to_csv(out, index=False)
    logging.info("Saved %s sweep results to %s", sweep_name, out)

def run_subset_experiment(lang, data_file, test_size, default_cfg):
    """
    For each subset CSV file in subsets/, train predictor and evaluate on test set.
    Result is one CSV of test metrics per subset.

    Args:
        lang: Language.
        data_file: Path to main dataset.
        test_size: Proportion for test split.
        default_cfg: Embedding and model settings.
    """
    subset_dir = "subsets"
    if not os.path.isdir(subset_dir):
        raise FileNotFoundError(f"No subset folder found at {subset_dir}")

    # Prepare test set (shared across all subsets)
    dp = DataProcessor(
        file_path   = data_file,
        word_column = 'word',
        score_column= 'concreteness',
        train_file  = f"{lang}_training.csv",
        test_file   = f"{lang}_test.csv"
    )
    _, test_set = dp.split_data(test_size=test_size,
                                random_state=42,
                                stratify=False)

    # Load embeddings (used by all runs)
    cfg = default_cfg.copy()
    emb = _load_embeddings(cfg)

    records = []
    for fn in sorted(os.listdir(subset_dir)):
        if not fn.endswith(".csv"):
            continue

        # Load this subset's data as training set
        train_df = pd.read_csv(os.path.join(subset_dir, fn))
        subset_size = len(train_df)

        # Train predictor and evaluate on the held-out test set
        pred = ConcretenessPredictor(embeddings=emb, k=cfg['k'], weighting=cfg['weighting'])
        ev   = Evaluator(predictor=pred)

        pred.fit(train_df)
        m = ev.evaluate_test_set(test_set)

        records.append({
            'subset_file':   fn,
            'subset_size':   subset_size,
            'rmse':          m['rmse'],
            'pearson_corr':  m['pearson_corr'],
            'spearman_corr': m['spearman_corr'],
        })

    # Write out one result CSV for all subsets
    result_dir = os.path.join("results", lang)
    os.makedirs(result_dir, exist_ok=True)
    out = os.path.join(
        result_dir,
        f"subset_experiment_{lang}_{datetime.now():%Y%m%d_%H%M}.csv"
    )
    pd.DataFrame(records).to_csv(out, index=False)
    logging.info("Saved subset experiment results to %s", out)

def run_per_word_cv(lang, data_file, test_size, default_cfg):
    """
    Run 5-fold cross-validation and write a per-word CSV with true and predicted concreteness scores for all words in all folds.

    Args:
        lang: Language.
        data_file: Path to main dataset.
        test_size: Proportion for test split.
        default_cfg: Model/embedding settings.
    """
    dp = DataProcessor(
        file_path=data_file,
        word_column='word',
        score_column='concreteness',
        train_file=f"{lang}_training.csv",
        test_file =f"{lang}_test.csv"
    )
    train_set, _ = dp.split_data(
        test_size=test_size,
        random_state=42,
        stratify=False
    )
    dp.train_set = train_set
    folds = list(dp.create_folds(n_splits=5, random_state=42, stratify=False))

    cfg  = default_cfg.copy()
    emb  = _load_embeddings(cfg)
    pred = ConcretenessPredictor(embeddings=emb, k=cfg['k'], weighting=cfg['weighting'])

    all_preds = []
    for fold_idx, (tr_idx, va_idx) in enumerate(folds, start=1):
        tr_df = train_set.iloc[tr_idx]
        va_df = train_set.iloc[va_idx].copy()

        pred.fit(tr_df)
        va_df['predicted'] = pred.predict(va_df)
        va_df['fold']      = fold_idx
        va_df['k']         = cfg['k']
        va_df['weighting'] = cfg['weighting']
        va_df['embedding'] = cfg['emb_path']

        all_preds.append(
            va_df[['word','concreteness','predicted','fold','k','weighting','embedding']]
        )

    per_word_cv = pd.concat(all_preds, axis=0).reset_index(drop=True)
    result_dir = os.path.join("results", lang)
    os.makedirs(result_dir, exist_ok=True)
    out = os.path.join(result_dir, f"per_word_cv_{lang}_{datetime.now():%Y%m%d_%H%M}.csv")
    per_word_cv.to_csv(out, index=False, encoding='utf-8-sig')
    logging.info("Saved per-word CV predictions to %s", out)


def final_test(lang, data_file, test_size, default_cfg, per_word):
    """
    Train predictor on full training set, evaluate on test set, write summary CSV and optionally per-word predictions.
    
    Args:
        lang: Language.
        data_file: Path to dataset.
        test_size: Test split size.
        default_cfg: Model/embedding settings.
        per_word: If True, output per-word predictions for test set.
    """
    dp = DataProcessor(file_path=data_file,
                       word_column='word',
                       score_column='concreteness',
                       train_file=f"{lang}_training.csv",
                       test_file =f"{lang}_test.csv")
    train_set, test_set = dp.split_data(test_size=test_size, random_state=42, stratify=False)

    cfg = default_cfg.copy()
    emb = _load_embeddings(cfg)
    pred = ConcretenessPredictor(embeddings=emb,
                                 k=cfg['k'],
                                 weighting=cfg['weighting'])
    ev = Evaluator(predictor=pred)

    pred.fit(train_set)
    m = ev.evaluate_test_set(test_set)

    rec = {
        **cfg,
        'train_size': len(train_set),
        'test_size':  test_size,
        'rmse':          m['rmse'],
        'pearson_corr':  m['pearson_corr'],
        'spearman_corr': m['spearman_corr'],    }
    df = pd.DataFrame([rec])
    result_dir = os.path.join("results", lang)
    os.makedirs(result_dir, exist_ok=True)
    out = os.path.join(result_dir, f"final_test_{lang}_{datetime.now():%Y%m%d_%H%M}.csv")
    df.to_csv(out, index=False)
    logging.info("Saved final test results to %s", out)

    if per_word:
        # Write per-word predictions for test set
        pw = test_set[['word','concreteness']].copy()
        pw['predicted'] = pred.predict(test_set)
        pw['k']         = cfg['k']
        pw['weighting'] = cfg['weighting']
        pw['embedding'] = cfg['emb_path']
        pw_out = os.path.join(result_dir, f"per_word_{lang}_{datetime.now():%Y%m%d_%H%M}.csv")
        pw.to_csv(pw_out, index=False)
        logging.info("Saved per-word predictions to %s", pw_out)


if __name__ == "__main__":
    # Parse command-line arguments for experiment configuration
    p = argparse.ArgumentParser(description="Concreteness experiments harness")
    p.add_argument('--language',     choices=['en','ar'], required=True,
                   help="Language: English (en) or Arabic (ar).")
    p.add_argument('--experiment',   choices=['k','weighting','embeddings','per_word_cv','subsets','final_test'],
                   help="Which sweep to run; defaults to final_test.")
    p.add_argument('--test_size',    type=float, default=None,
                   help="Override test split (en↦0.1, ar↦0.2 if unset).")
    p.add_argument('--n_folds',      type=int,   default=5,
                   help="Number of CV folds.")
    p.add_argument('--per_word',     action='store_true',
                   help="On final_test, also output per-word CSV.")
    # Experiment and model hyperparameter overrides
    p.add_argument('--k',            type=int,
                   help="Override number of neighbors (default per-language).")
    p.add_argument('--weighting',    choices=['uniform','distance'],
                   help="Override weighting scheme.")
    p.add_argument('--embedding_type', choices=['fasttext','transformer'],
                   default='fasttext',
                   help="Choose fasttext or transformer static embeddings.")
    p.add_argument('--model_variant', choices=['bert','gpt2','roberta'],
                   default='bert',
                   help="For transformer: which variant.")
    p.add_argument('--static_mode',  choices=['sent','para'],
                   default='sent',
                   help="For transformer: para vs. sent distilled.")
    p.add_argument('--model_path',   type=str,
                   help="Override embedding file path.")
    p.add_argument('--file_format',  choices=['bin','vec'],
                   help="Override fasttext file format.")
    args = p.parse_args()

    # Set default test split depending on language
    if args.test_size is None:
        args.test_size = 0.1 if args.language=='en' else 0.2

    # Set per-language default configs
    if args.language == 'en':
        data_file = 'datasets/en/english_dataset.xlsx'
        default_cfg = {
            'k': 35, 'weighting': 'distance',
            'emb_type': 'fasttext',
            'emb_path': 'crawl-300d-2M-subword.bin',
            'emb_fmt': 'bin'
        }
    else:
        data_file = 'datasets/ar/arabic_dataset.xlsx'
        default_cfg = {
            'k': 20, 'weighting': 'distance',
            'emb_type': 'transformer',
            'emb_path': 'ar_gpt2_sent.vec',
            'emb_fmt': 'vec'
        }

    # Apply CLI overrides to config if provided
    default_cfg = _derive_embedding_cfg(args, default_cfg)

    # Choose experiment type and dispatch to appropriate handler
    exp = args.experiment or 'final_test'
    if exp == 'k':
        # Sweep k hyperparameter
        grid = [ {**default_cfg, 'k': k} for k in range(5, 101, 5) ]
        run_experiment_sweep(args.language, data_file, args.test_size, args.n_folds,
                             default_cfg, 'k_experiment', grid)

    elif exp == 'per_word_cv':
        run_per_word_cv(
            lang       = args.language,
            data_file  = data_file,
            test_size  = args.test_size,
            default_cfg= default_cfg
        )

    elif exp == 'subsets':
        run_subset_experiment(
            lang        = args.language,
            data_file   = data_file,
            test_size   = args.test_size,
            default_cfg = default_cfg
        )

    elif exp == 'weighting':
        # Sweep weighting hyperparameter
        grid = [ {**default_cfg, 'weighting': w} for w in ('uniform','distance') ]
        run_experiment_sweep(args.language, data_file, args.test_size, args.n_folds,
                             default_cfg, 'weighting_experiment', grid)

    elif exp == 'embeddings':
        # Sweep over multiple embedding files
        if args.language=='en':
            emb_files = [
                ('fasttext','crawl-300d-2M-subword.bin','bin'),
                ('fasttext','crawl-300d-2M.vec','vec'),
                ('fasttext','wiki-news-300d-1M-subword.vec','vec'),
                ('fasttext','wiki-news-300d-1M.vec','vec'),
                ('transformer','bert_24layer_para.vec','vec'),
                ('transformer','bert_24layer_sent.vec','vec'),
                ('transformer','roberta_24layer_para.vec','vec'),
                ('transformer','roberta_24layer_sent.vec','vec'),
                ('transformer','GPT2_24_layer_para.vec','vec'),
                ('transformer','GPT2_24_layer_sent.vec','vec'),
            ]
        else:
            emb_files = [
                ('fasttext','wiki.ar.bin','bin'),
                ('fasttext','cc.ar.300.bin','bin'),
                ('transformer','ar_bert_sent.vec','vec'),
                ('transformer','ar_gpt2_sent.vec','vec'),
                ('transformer','ar_roberta_sent.vec','vec'),
            ]
        emb_dir = os.path.join(EMBEDDINGS_ROOT, args.language)
        grid = []
        # Build a grid of embedding configs for the sweep
        for (t, fname, fmt) in emb_files:
            full_path = (fname if os.path.isabs(fname)
                         else os.path.join(emb_dir, fname))
            grid.append({
                **default_cfg,
                'emb_type': t,
                'emb_path': full_path,
                'emb_fmt':  fmt
            })
        run_experiment_sweep(
            args.language, 
            data_file, 
            args.test_size, 
            args.n_folds,
            default_cfg, 
            'embedding_experiment', 
            grid
        )

    else:  # Default case: final_test
        final_test(
            args.language, 
            data_file, 
            args.test_size,
            default_cfg, 
            args.per_word
        )
