# Plasma Arch Prediction

## Data Preprocessing Pipeline
1. Get Flags
```
self.sys_flags = flags_module.get_sys_flags()
self.data_flags = flags_module.get_data_flags()
self.train_flags = flags_module.get_data_flags()
```
2. Check preprocessed_dir
```
dir_processed = os.path.join(dir_processed, 'Ls_%d_Ls_shift_%d')
```
2. Transform Data from TXT to Numpy


## Training PipeLine
