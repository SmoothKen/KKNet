### KKNet

An implementation of "[Towards Improving Harmonic Sensitivity and Prediction Stability for Singing Melody Extraction](https://arxiv.org/abs/2308.02723)", in ISMIR 2023

Will update training/inference instructions soon. Basically ``python feature_extraction.py`` for caching CFP/z-CFP before training. Then ``python main.py train`` will call ``tonet.py`` and start the main training loop. ``tonet.py`` in turn calls the PianoNet model in ``piano_net.py``.

Standalone testing can be done using ``python main.py test``

The data used for the experiments can be found here: https://drive.google.com/file/d/1QKX6rpuRxMPt54HOqNQztmLQqGlCALZ4/view?usp=sharing

