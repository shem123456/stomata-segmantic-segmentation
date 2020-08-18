# stomata-segmantic-segmentation
a method for observing and analyzing patchy stomata in wheat leaves based on object tracking and semantic segmentation

## Requirements<p>
tensorflow-gpu == 1.13.1 keras == 2.1.5 opencv-python == 4.2.0 numpy == 1.18.0<p>
  
## data
Dataset1 contains three files: jpg, png, train.txt.<p>
  jpg stores original training images.
  png stores label images.
  train.txt provides the path of images.
  
## train
when you want to train your own model, you need run train.py. Trained model will svaed in logs.

## test
The images to test are stored on the img, and you run predict.py. The results will be saved in img_out.<p>
  
If you want to test the video file, you should run tracing_with_segmention.py. Firstly, the demo will remind you to mark three boxes, and then, the demo will trace your marked stomata and segment the stomatal aperture. Finally, a video will be saved as the result.

