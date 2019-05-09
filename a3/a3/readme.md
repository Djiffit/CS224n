Epoch 10 out of 10
100%|███████████████████████████████████████████████████████████████████████████████████████████████████| 1848/1848 [02:08<00:00, 14.35it/s]
Average Train Loss: 0.057947821397076306
Evaluating on dev set
1445850it [00:00, 57591567.24it/s]                                                                                                          
- dev UAS: 88.67
New best dev UAS! Saving model.

================================================================================
TESTING
================================================================================
Restoring the best model weights found on the dev set
Final evaluation on test set
2919736it [00:00, 83837725.38it/s]                                                                                                          
- test UAS: 89.09
Done!

