# Plasma Arch Prediction

## Data Preprocessing Pipeline
1. Get Flags
```
self.sys_flags = flags_module.get_sys_flags()
self.data_flags = flags_module.get_data_flags()
self.train_flags = flags_module.get_data_flags()
```
2. factory_class
```
self.prepare_dirs()
self.txt_to_npy()
self.make_npy_figs() : For the Data Check
```
2. Transform Data from TXT to Numpy
```
dir_npys = factory_module.txt_to_npy(dir_processed, dir_raw_data)
```



## Training PipeLine
