# ndvi

```bash
ubuntu@ubuntu:~/ndvi$ chmod a+x ndvi.py && ./ndvi.py --help
usage: ndvi.py [-h] -n NIR -v VIS -nt NIR_TRANSFORM -vt VIS_TRANSFORM
               [-p {ndvi,rvi}] [-s SAVE_NAME]

optional arguments:
  -h, --help            show this help message and exit
  -n NIR, --nir NIR     NIR band image directory
  -v VIS, --vis VIS     vis band image directory
  -nt NIR_TRANSFORM, --nir_transform NIR_TRANSFORM
                        NIR transform array file directory
  -vt VIS_TRANSFORM, --vis_transform VIS_TRANSFORM
                        vis transform array file directory
  -p {ndvi,rvi}, --pattern {ndvi,rvi}
                        calculate ndvi or rvi
  -s SAVE_NAME, --save_name SAVE_NAME
                        ndvi image save name
```

