# Picture Inpainting

1. Python implementation of the paper [IMAGE INPAINTING VIA SPARSE REPRESENTATION ](<<https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4959679>>), ICASSP 2009

2. Picture inpainting using Python Package [Dictlearn](<https://dictlearn.readthedocs.io/en/latest/algorithms.html#inpaint>).

## Paper based method

### Run

To run the reimplemented project for the paper, place your original image and corresponding mask in the path `./pictures` like this:

```
-pictures
	-image file
	-image_mask file
```

And then run:

 ```sh
 python run_paper.py
 ```

### Options

`--picture_path` : path to the original picture
`--mask_path`       : path to the mask, it has to be black and white, the black part indicates the missing pixels
`--ave_dir`           : path to save the results
`--patch_size`     : side-length for a square patch
`--step`                  : step interval to fetch the patches
`--alpha`                : alpha value for Lasso
`--max_iter`          : max_iteration time for Lasso
`--tolerance`        : tolerance value for Lasso
`--local`                : whether to build the dictionary locally

### Results

The results will be save at `./results`,including the inpainted image and a report for a single experiment will be save at `./results/results.txt`. An example is like below:

```txt
>>> Experiment Time: 20190610-193633 
>>> Experiment Settings: 
patch_size: 101 | step: 25 | alpha: 0.001000 | tolerance: 0.000100 | max_iter: 10000 | local: 1 
>>>Experimental Attribute: 
total patch num: 470 | total missing pixel num: 11414 | average iteration: 100 | total time used: 293 s 
>>>Experimental Metrics: 
MSE: 0.002956 | PSNR: 25.293428 | SSIM: 0.957037 >>> Inpainted picture save at: 
./results/hill_inpaint_20190610-193633.jpg 
```



## Dictlearn based method

### Run

To run the `dictlearn` based  method, place your original image and corresponding mask in the path `./pictures` the same as above:

```
-pictures
	-image file
	-image_mask file
```

And then run:

```sh
  python run_package.py
```

### Results

The results will be save at `./results/package/`



------

Should you have any advice on the project or any problem using it, feel free to let me know. Issues are welcome!