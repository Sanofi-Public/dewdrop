#!/bin/bash 

# dewdrop
python dewdrop_synth.py --ensemble_size 128 --selection_rounds 10 -v 1 --selection_batch_size 100
# python dewdrop_synth.py --ensemble_size 128 --selection_rounds 10 -v 2 --selection_batch_size 100
# python dewdrop_synth.py --ensemble_size 128 --selection_rounds 10 -v 3 --selection_batch_size 100

# badge 
python badge_synth.py --selection_rounds 10 -v 1 --selection_batch_size 100
# python badge_synth.py --selection_rounds 20 -v 2 --selection_batch_size 100
# python badge_synth.py --selection_rounds 20 -v 3 --selection_batch_size 100

# # kmeans 
python kmeans_synth.py --selection_rounds 10 -v 1 --selection_batch_size 100
# python kmeans_synth.py --selection_rounds 20 -v 2 --selection_batch_size 100
# python kmeans_synth.py --selection_rounds 20 -v 3 --selection_batch_size 100

# # random 
python random_synth.py --selection_rounds 10 -v 1 --selection_batch_size 100
# python random_synth.py --selection_rounds 20 -v 2 --selection_batch_size 100
# python random_synth.py --selection_rounds 20 -v 3 --selection_batch_size 100