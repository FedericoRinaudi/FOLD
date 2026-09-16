"""FOLD: website fingerprinting on Oblivious DNS over HTTPS.

    fold.data          loading, splitting, the defense, and the encoder that
                       turns traces into the inputs each model reads
    fold.models        the attack and the baselines
    fold.training      building, compiling and fitting them
    fold.predictions   what a model leaves on disk, and how to read it back
    fold.metrics       the metrics of the paper
    fold.experiments   the experiments, one function per model and scenario
    fold.background    the synthetic open-world background
"""
