# VGG Boxes to Mask

Create a set of GIF masks from a [VGG Image Annotator (VIA)](https://annotate.officialstatistics.org/) project JSON file. A GIF is created for each annotated image in the project and each frame in each GIF is a mask for a different class. The program can also create coloured PNG masks to help visualise the masks, this is achieved by running the program without the gif (-g) tag. As this program was created for my project's specific needs it has a few limitations that may need to be worked around if this is used in a different context.

1. This script is only compatible with rectangular bounding boxes so annotations of any other shape won't work.
2. All annotated images must also be of the same dimensions and these dimensions are hard-coded.
3. The class labels and class colours are all hard coded for sea ice semantic segmentation.

This script was created for my final year project in my degree, which was concerned with performing Semnatic Segmentation on images of sea ice.

## How to Use
Run `pipeline.sh` with the arguments listed in comments at the start of that file.
