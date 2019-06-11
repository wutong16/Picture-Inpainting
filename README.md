# Picture Inpainting

1. Python implementation of the paper [IMAGE INPAINTING VIA SPARSE REPRESENTATION ](<<https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4959679>>), ICASSP 2009

2. Picture inpainting using Python Package [Dictlearn](<https://dictlearn.readthedocs.io/en/latest/algorithms.html#inpaint>).

## Paper based method

To run the reimplemented project for the paper, place your original image and corresponding mask in the path `./pictures` like this:

```
-pictures
	-image.jpg/png/...
	-image_mask.jpg/png/...
```

And then run:

 ```sh
 python run_paper.py
 ```

#### Options

`--picture_path` : path to the original picture
`--mask_path`       : path to the mask, it has to be black and white, the black part indicates the missing pixels
`--ave_dir`           : path to save the results
`--patch_size`     : side-length for a square patch
`--step`                  : step interval to fetch the patches
`--alpha`                : alpha value for Lasso
`--max_iter`          : max_iteration time for Lasso
`--tolerance`        : tolerance value for Lasso
`--local`                : whether to build the dictionary locally

## Dictlearn based method

To run the `dictlearn` based  method, place your original image and corresponding mask in the path `./pictures` the same as above:

```
-pictures
	-image.jpg/png/...
	-image_mask.jpg/png/...
```

And then run:

```sh
  python run_package.py
```


